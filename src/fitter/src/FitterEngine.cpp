//
// Created by Adrien BLANCHET on 11/06/2021.
//

#include <Math/Factory.h>
#include "TGraph.h"

#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"

#include "JsonUtils.h"
#include "FitterEngine.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[FitterEngine]");
})

FitterEngine::FitterEngine() { this->reset(); }
FitterEngine::~FitterEngine() { this->reset(); }

void FitterEngine::reset() {
  _saveDir_ = nullptr;
  _config_.clear();

  _propagator_.reset();
  _minimizer_.reset();
  _functor_.reset();
  _nbFitParameters_ = 0;
  _nbFitCalls_ = 0;

  _convergenceMonitor_.reset();
}

void FitterEngine::setSaveDir(TDirectory *saveDir) {
  _saveDir_ = saveDir;
}
void FitterEngine::setConfig(const nlohmann::json &config_) {
  _config_ = config_;
  while( _config_.is_string() ){
    LogWarning << "Forwarding " << __CLASS_NAME__ << " config: \"" << _config_.get<std::string>() << "\"" << std::endl;
    _config_ = JsonUtils::readConfigFile(_config_.get<std::string>());
  }
}

void FitterEngine::initialize() {

  if( _config_.empty() ){
    LogError << "Config is empty." << std::endl;
    throw std::runtime_error("config not set.");
  }

  initializePropagator();
  initializeMinimizer();

  _convergenceMonitor_.addDisplayedQuantity("VarName");
  _convergenceMonitor_.addDisplayedQuantity("LastAddedValue");
  _convergenceMonitor_.addDisplayedQuantity("SlopePerCall");

  _convergenceMonitor_.getQuantity("VarName").title = "Chi2";
  _convergenceMonitor_.getQuantity("LastAddedValue").title = "Current Value";
  _convergenceMonitor_.getQuantity("SlopePerCall").title = "Avg. Slope /call";

  _convergenceMonitor_.addVariable("Total");
  _convergenceMonitor_.addVariable("Stat");
  _convergenceMonitor_.addVariable("Syst");

}

void FitterEngine::generateSamplePlots(const std::string& saveDir_){

  LogInfo << __METHOD_NAME__ << std::endl;

  _propagator_.propagateParametersOnSamples();
  _propagator_.fillSampleHistograms();
  _propagator_.getPlotGenerator().generateSamplePlots(
    GenericToolbox::mkdirTFile(_saveDir_, saveDir_ )
    );

}
void FitterEngine::generateOneSigmaPlots(const std::string& saveDir_){

  _propagator_.propagateParametersOnSamples();
  _propagator_.fillSampleHistograms();
  _propagator_.getPlotGenerator().generateSamplePlots();

  _saveDir_->cd(); // to put this hist somewhere
  auto refHistList = _propagator_.getPlotGenerator().getHistHolderList(); // current buffer

  // +1 sigma
  int iPar = -1;
  for( auto& parSet : _propagator_.getParameterSetsList() ){
    for( auto& par : parSet.getParameterList() ){
      iPar++;

      std::string tag;
      if( _minimizer_->IsFixedVariable(iPar) ){
        tag += "_FIXED";
//        LogInfo << "Not processing fixed parameter: " << parSet.getName() + "/" + par.getTitle() << std::endl;
//        continue;
      }

      double currentParValue = par.getParameterValue();
      par.setParameterValue( currentParValue + par.getStdDevValue() );
      LogInfo << "(" << iPar+1 << "/" << _nbFitParameters_ << ") +1 sigma on " << parSet.getName() + "/" + par.getTitle()
      << " -> " << par.getParameterValue() << std::endl;
      _propagator_.propagateParametersOnSamples();
      _propagator_.fillSampleHistograms();

      std::string savePath = saveDir_;
      if( not savePath.empty() ) savePath += "/";
      savePath += "oneSigma/" + parSet.getName() + "/" + par.getTitle() + tag;
      auto* saveDir = GenericToolbox::mkdirTFile(_saveDir_, savePath );
      saveDir->cd();

      _propagator_.getPlotGenerator().generateSamplePlots();

      auto oneSigmaHistList = _propagator_.getPlotGenerator().getHistHolderList();
      _propagator_.getPlotGenerator().generateComparisonPlots( oneSigmaHistList, refHistList, saveDir );
      par.setParameterValue( currentParValue );
      _propagator_.propagateParametersOnSamples();
      _propagator_.fillSampleHistograms();

      const auto& compHistList = _propagator_.getPlotGenerator().getComparisonHistHolderList();
//      bool isAffected = false;
//      for( const auto& compHist : compHistList ){
//        for( int iBin = 1 ; iBin <= compHist.histPtr->GetNbinsX() ; iBin++ ){
//          if( TMath::Abs( compHist.histPtr->GetBinContent(iBin) ) > 0.1 ){ // 0.1 % do decide if
//            isAffected = true;
//            break;
//          }
//        }
//        if( isAffected ) break;
//      }
//      if( not isAffected ){
//        LogWarning << parSet.getName() + "/" + par.getTitle() << " has no effect on the sample. Fixing in the fit." << std::endl;
//        _minimizer_->FixVariable(iPar);
//        par.setIsFixed(true); // ignored in the Chi2 computation of the parSet
//      }

      // Since those were not saved, delete manually
      for( auto& hist : oneSigmaHistList ){ delete hist.histPtr; }
      oneSigmaHistList.clear();
    }
  }

  _saveDir_->cd();

  // Since those were not saved, delete manually
  for( auto& refHist : refHistList ){ delete refHist.histPtr; }
  refHistList.clear();

}

void FitterEngine::fixGhostParameters(){
  LogInfo << __METHOD_NAME__ << std::endl;

  _propagator_.propagateParametersOnSamples();
  _propagator_.fillSampleHistograms();
  updateChi2Cache();

  double baseChi2Stat = _chi2StatBuffer_;

  // +1 sigma
  int iPar = -1;
  for( auto& parSet : _propagator_.getParameterSetsList() ){
    for( auto& par : parSet.getParameterList() ){
      iPar++;

      double currentParValue = par.getParameterValue();
      par.setParameterValue( currentParValue + par.getStdDevValue() );
      LogInfo << "(" << iPar+1 << "/" << _nbFitParameters_ << ") +1 sigma on " << parSet.getName() + "/" + par.getTitle()
              << " -> " << par.getParameterValue() << std::endl;
      _propagator_.propagateParametersOnSamples();
      _propagator_.fillSampleHistograms();

      // Compute the Chi2
      updateChi2Cache();

      double deltaChi2 = _chi2StatBuffer_ - baseChi2Stat;
      LogDebug << GET_VAR_NAME_VALUE(deltaChi2) << std::endl;

      if( std::fabs(deltaChi2) < 1E-6 ){
        LogWarning << parSet.getName() + "/" + par.getTitle() << " has no effect on the sample. Fixing in the fit." << std::endl;
        _minimizer_->FixVariable(iPar);
        par.setIsFixed(true); // ignored in the Chi2 computation of the parSet
      }

      par.setParameterValue( currentParValue );
      _propagator_.propagateParametersOnSamples();

    }
  }
}
void FitterEngine::scanParameters(int nbSteps_, const std::string &saveDir_) {
  LogInfo << "Performing parameter scans..." << std::endl;
  for( int iPar = 0 ; iPar < _minimizer_->NDim() ; iPar++ ){
    this->scanParameter(iPar, nbSteps_, saveDir_);
  } // iPar
}
void FitterEngine::scanParameter(int iPar, int nbSteps_, const std::string &saveDir_) {

  //Internally Scan performs steps-1, so add one to actually get the number of steps
  //we ask for.
  unsigned int adj_steps = nbSteps_+1;
  auto* x = new double[adj_steps] {};
  auto* y = new double[adj_steps] {};

  LogInfo << "Scanning fit parameter #" << iPar
          << " (" << _minimizer_->VariableName(iPar) << ")." << std::endl;

  bool success = _minimizer_->Scan(iPar, adj_steps, x, y);

  if( not success ){
    LogError << "Parameter scan failed." << std::endl;
  }

  TGraph scanGraph(nbSteps_, x, y);

  std::stringstream ss;
  ss << GenericToolbox::replaceSubstringInString(_minimizer_->VariableName(iPar), "/", "_");
  ss << "_TGraph";

  scanGraph.SetTitle(_minimizer_->VariableName(iPar).c_str());

  if( _saveDir_ != nullptr ){
    GenericToolbox::mkdirTFile(_saveDir_, saveDir_)->cd();
    scanGraph.Write( ss.str().c_str() );
  }

  delete[] x;
  delete[] y;
}

void FitterEngine::fit(){
  LogAlert << __METHOD_NAME__ << std::endl;

  bool _fitHasConverged_ = _minimizer_->Minimize();
  if( _fitHasConverged_ ){
    LogInfo << "Fit converged." << std::endl
            << "Status code: " << _minimizer_->Status() << std::endl;

    LogInfo << "Calling HESSE." << std::endl;
    _fitHasConverged_ = _minimizer_->Hesse();

    if(not _fitHasConverged_){
      LogError  << "Hesse did not converge." << std::endl;
      LogError  << "Failed with status code: " << _minimizer_->Status() << std::endl;
    }
    else{
      LogInfo << "Hesse converged." << std::endl
              << "Status code: " << _minimizer_->Status() << std::endl;
    }

  }
  else{
    LogError << "Did not converged." << std::endl;
  }

  LogDebug << _convergenceMonitor_.generateMonitorString(); // lasting printout

}
void FitterEngine::updateChi2Cache(){

  ////////////////////////////////
  // Compute chi2 stat
  ////////////////////////////////
  _chi2StatBuffer_ = 0; // reset
  double buffer;
  for( const auto& sample : _propagator_.getSamplesList() ){

    //buffer = _samplesList_.at(sampleContainer)->CalcChi2();
    buffer = sample.CalcLLH();
    //buffer = _samplesList_.at(sampleContainer)->CalcEffLLH();

    _chi2StatBuffer_ += buffer;
  }


  ////////////////////////////////
  // Compute the penalty terms
  ////////////////////////////////
  _chi2PullsBuffer_ = 0;
  _chi2RegBuffer_ = 0;
  for( auto& parSet : _propagator_.getParameterSetsList() ){
    _chi2PullsBuffer_ += parSet.getChi2();
  }

  _chi2Buffer_ = _chi2StatBuffer_ + _chi2PullsBuffer_ + _chi2RegBuffer_;

}
double FitterEngine::evalFit(const double* parArray_){

  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  _nbFitCalls_++;

  // Update fit parameter values:
  int iPar = -1;
  for( auto& parSet : _propagator_.getParameterSetsList() ){
    for( auto& par : parSet.getParameterList() ){
      iPar++;
      par.setParameterValue( parArray_[iPar] );
    }
  }

  _propagator_.propagateParametersOnSamples();
  _propagator_.fillSampleHistograms();

  // Compute the Chi2
  updateChi2Cache();

  _convergenceMonitor_.setHeaderString(__METHOD_NAME__ + " took: " + GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__));
  _convergenceMonitor_.getVariable("Total").addQuantity(_chi2Buffer_);
  _convergenceMonitor_.getVariable("Stat").addQuantity(_chi2StatBuffer_);
  _convergenceMonitor_.getVariable("Syst").addQuantity(_chi2PullsBuffer_);
  LogDebug << _convergenceMonitor_.generateMonitorString(true);


//  LogDebug.clearLine();
//  LogDebug << __METHOD_NAME__ << " took: " << GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__) << std::endl;
//  LogDebug << GET_VAR_NAME_VALUE(_chi2Buffer_) << std::endl;
//  LogDebug << GET_VAR_NAME_VALUE(_chi2StatBuffer_) << std::endl;
//  LogDebug << GET_VAR_NAME_VALUE(_chi2PullsBuffer_) << std::endl;
//  LogDebug.moveTerminalCursorBack(4);

  return _chi2Buffer_;
}


void FitterEngine::initializePropagator(){

  _propagator_.setConfig(JsonUtils::fetchValue<json>(_config_, "propagatorConfig"));

  TFile* f = TFile::Open(JsonUtils::fetchValue<std::string>(_config_, "mc_file").c_str(), "READ");
  _propagator_.setDataTree( f->Get<TTree>("selectedEvents") );
  _propagator_.setMcFilePath(JsonUtils::fetchValue<std::string>(_config_, "mc_file"));

  if( _saveDir_ != nullptr ){
    _propagator_.setSaveDir(GenericToolbox::mkdirTFile(_saveDir_, "propagator"));
  }

  _propagator_.initialize();

  LogTrace << "Counting parameters" << std::endl;
  _nbFitParameters_ = 0;
  for( const auto& parSet : _propagator_.getParameterSetsList() ){
    _nbFitParameters_ += int(parSet.getNbParameters());
  }
  LogTrace << GET_VAR_NAME_VALUE(_nbFitParameters_) << std::endl;

}
void FitterEngine::initializeMinimizer(){

  auto minimizationConfig = JsonUtils::fetchSubEntry(_config_, {"minimizerConfig"});
  if( minimizationConfig.is_string() ){ minimizationConfig = JsonUtils::readConfigFile(minimizationConfig.get<std::string>()); }

  _minimizer_ = std::shared_ptr<ROOT::Math::Minimizer>(
    ROOT::Math::Factory::CreateMinimizer(
      JsonUtils::fetchValue<std::string>(minimizationConfig, "minimizer"),
      JsonUtils::fetchValue<std::string>(minimizationConfig, "algorithm")
    )
  );

  _functor_ = std::shared_ptr<ROOT::Math::Functor>(
    new ROOT::Math::Functor(
      this, &FitterEngine::evalFit, _nbFitParameters_
      )
  );

  _minimizer_->SetFunction(*_functor_);
  _minimizer_->SetStrategy(JsonUtils::fetchValue<int>(minimizationConfig, "strategy"));
  _minimizer_->SetPrintLevel(JsonUtils::fetchValue<int>(minimizationConfig, "print_level"));
  _minimizer_->SetTolerance(JsonUtils::fetchValue<double>(minimizationConfig, "tolerance"));
  _minimizer_->SetMaxIterations(JsonUtils::fetchValue<unsigned int>(minimizationConfig, "max_iter"));
  _minimizer_->SetMaxFunctionCalls(JsonUtils::fetchValue<unsigned int>(minimizationConfig, "max_fcn"));

  int iPar = -1;
  for( auto& parSet : _propagator_.getParameterSetsList() ){

    for( auto& par : parSet.getParameterList()  ){
      iPar++;

      _minimizer_->SetVariable(
        iPar,
        parSet.getName() + "/" + par.getTitle(),
        par.getParameterValue(),
        0.01
      );
    } // par
  } // parSet

  LogInfo << "Number of defined parameters: " << _minimizer_->NDim() << std::endl
          << "Number of free parameters   : " << _minimizer_->NFree() << std::endl
          << "Number of fixed parameters  : " << _minimizer_->NDim() - _minimizer_->NFree()
          << std::endl;

}
