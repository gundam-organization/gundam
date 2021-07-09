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
  _fitIsDone_ = false;
  _saveDir_ = nullptr;
  _config_.clear();
  _chi2History_.clear();

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

  _propagator_.propagateParametersOnEvents();
  _propagator_.fillSampleHistograms();
  _propagator_.getPlotGenerator().generateSamplePlots(
    GenericToolbox::mkdirTFile(_saveDir_, saveDir_ )
    );

}
void FitterEngine::generateOneSigmaPlots(const std::string& saveDir_){

  _propagator_.propagateParametersOnEvents();
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
      if( _minimizer_->IsFixedVariable(iPar) ){ tag += "_FIXED"; }

      double currentParValue = par.getParameterValue();
      par.setParameterValue( currentParValue + par.getStdDevValue() );
      LogInfo << "(" << iPar+1 << "/" << _nbFitParameters_ << ") +1 sigma on " << parSet.getName() + "/" + par.getTitle()
      << " -> " << par.getParameterValue() << std::endl;
      _propagator_.propagateParametersOnEvents();
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
      _propagator_.propagateParametersOnEvents();
      _propagator_.fillSampleHistograms();

      const auto& compHistList = _propagator_.getPlotGenerator().getComparisonHistHolderList();

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

  _propagator_.propagateParametersOnEvents();
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
      _propagator_.propagateParametersOnEvents();
      _propagator_.fillSampleHistograms();

      // Compute the Chi2
      updateChi2Cache();

      double deltaChi2 = _chi2StatBuffer_ - baseChi2Stat;

      if( std::fabs(deltaChi2) < 1E-6 ){
        LogAlert << parSet.getName() + "/" + par.getTitle() << ": Δχ² = " << deltaChi2 << " < " << 1E-6 << ", fixing parameter." << std::endl;
        _minimizer_->FixVariable(iPar);
        par.setIsFixed(true); // ignored in the Chi2 computation of the parSet
      }

      par.setParameterValue( currentParValue );
      _propagator_.propagateParametersOnEvents();

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

void FitterEngine::throwParameters(){
  LogInfo << __METHOD_NAME__ << std::endl;

  int iPar = 0;
  for( auto& parSet : _propagator_.getParameterSetsList() ){
    for( auto& par : parSet.getParameterList() ){
      if( par.isEnabled() and not par.isFixed() ){
        par.setParameterValue( par.getPriorValue() + _prng_.Gaus(0, par.getStdDevValue()) );
        _minimizer_->SetVariableValue( iPar, par.getParameterValue() );
      }
    }
    iPar++;
  }

  _propagator_.propagateParametersOnEvents();
  _propagator_.fillSampleHistograms();

}

void FitterEngine::fit(){
  LogWarning << __METHOD_NAME__ << std::endl;

  for( const auto& parSet : _propagator_.getParameterSetsList() ){
    LogWarning << parSet.getName() << ": " << parSet.getNbParameters() << " parameters" << std::endl;
    for( const auto& par : parSet.getParameterList() ){
      if( par.isEnabled() ){
        if( par.isFixed() ){
          LogAlert << "\033[41m" << parSet.getName() << "/" << par.getTitle() << ": FIXED - Prior: " << par.getParameterValue() <<  "\033[0m" << std::endl;
        }
        else if( not par.isEnabled() ){
          LogInfo << "\033[40m" << parSet.getName() << "/" << par.getTitle() << ": Disabled" <<  "\033[0m" << std::endl;
        }
        else{
          LogInfo << parSet.getName() << "/" << par.getTitle() << " - Prior: " << par.getParameterValue() << std::endl;
        }
      }
    }
  }

  _fitUnderGoing_ = true;
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

  _fitUnderGoing_ = false;
  _fitIsDone_ = true;
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

  _propagator_.propagateParametersOnEvents();
  _propagator_.fillSampleHistograms();

  // Compute the Chi2
  updateChi2Cache();

  if( _fitUnderGoing_ ){
    std::stringstream ss;
    ss << __METHOD_NAME__ << ": call #" << _nbFitCalls_ << std::endl;
    ss << "Computation time: " << GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__);
    _convergenceMonitor_.setHeaderString(ss.str());
    _convergenceMonitor_.getVariable("Total").addQuantity(_chi2Buffer_);
    _convergenceMonitor_.getVariable("Stat").addQuantity(_chi2StatBuffer_);
    _convergenceMonitor_.getVariable("Syst").addQuantity(_chi2PullsBuffer_);
    LogDebug << _convergenceMonitor_.generateMonitorString(true);

    // Fill History
    _chi2History_["Total"].emplace_back(_chi2Buffer_);
    _chi2History_["Stat"].emplace_back(_chi2StatBuffer_);
    _chi2History_["Syst"].emplace_back(_chi2PullsBuffer_);
  }

  return _chi2Buffer_;
}

void FitterEngine::writePostFitData() {
  LogInfo << __METHOD_NAME__ << std::endl;

  if( not _fitIsDone_ ){
    LogError << "Can't do " << __METHOD_NAME__ << " while fit has not been called." << std::endl;
    throw std::logic_error("Can't do " + __METHOD_NAME__ + " while fit has not been called.");
  }

  if( _saveDir_ != nullptr ){
    auto* postFitDir = GenericToolbox::mkdirTFile(_saveDir_, "postFit");

    GenericToolbox::mkdirTFile(postFitDir, "chi2")->cd();
    GenericToolbox::convertTVectorDtoTH1D(_chi2History_["Total"], "#chi^{2}(Total) - Converging history")->Write("chi2TotalHistory");
    GenericToolbox::convertTVectorDtoTH1D(_chi2History_["Stat"], "#chi^{2}(Stat) - Converging history")->Write("chi2StatHistory");
    GenericToolbox::convertTVectorDtoTH1D(_chi2History_["Syst"], "#chi^{2}(Syst) - Converging history")->Write("chi2PullsHistory");

    this->generateSamplePlots("postFit/samples");

    auto* errorDir = GenericToolbox::mkdirTFile(postFitDir, "errors");
//    const unsigned int nfree = _minimizer_->NFree();
    if(_minimizer_->X() != nullptr){
      double covarianceMatrixArray[_minimizer_->NDim() * _minimizer_->NDim()];
      _minimizer_->GetCovMatrix(covarianceMatrixArray);
      TMatrixDSym fitterCovarianceMatrix(_minimizer_->NDim(), covarianceMatrixArray);
      TH2D* fitterCovarianceMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) &fitterCovarianceMatrix, "fitterCovarianceMatrixTH2D");

      std::vector<double> parameterValueList(_minimizer_->X(),      _minimizer_->X()      + _minimizer_->NDim());
      std::vector<double> parameterErrorList(_minimizer_->Errors(), _minimizer_->Errors() + _minimizer_->NDim());

      int parameterIndexOffset = 0;
      for( const auto& parSet : _propagator_.getParameterSetsList() ){

        if( not parSet.isEnabled() ){
          continue;
        }

        auto* parSetDir = GenericToolbox::mkdirTFile(errorDir, parSet.getName());

        auto* covMatrix = new TMatrixD(parSet.getParameterList().size(), parSet.getParameterList().size());
        for( const auto& parRow : parSet.getParameterList() ){
          for( const auto& parCol : parSet.getParameterList() ){
            (*covMatrix)[ parRow.getParameterIndex() ][ parCol.getParameterIndex() ] =
              fitterCovarianceMatrix[parameterIndexOffset + parRow.getParameterIndex() ][parameterIndexOffset + parCol.getParameterIndex() ];
          } // par Y
        } // par X
        auto* covMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) covMatrix, Form("Covariance_%s_TH2D", parSet.getName().c_str()));

        auto* corMatrix = GenericToolbox::convertToCorrelationMatrix((TMatrixD*) covMatrix);
        auto* corMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D(corMatrix, Form("Correlation_%s_TH2D", parSet.getName().c_str()));

        for( const auto& par : parSet.getParameterList() ){
          covMatrixTH2D->GetXaxis()->SetBinLabel(1+par.getParameterIndex(), (parSet.getName() + "/" + par.getTitle()).c_str());
          covMatrixTH2D->GetYaxis()->SetBinLabel(1+par.getParameterIndex(), (parSet.getName() + "/" + par.getTitle()).c_str());
          corMatrixTH2D->GetXaxis()->SetBinLabel(1+par.getParameterIndex(), (parSet.getName() + "/" + par.getTitle()).c_str());
          corMatrixTH2D->GetYaxis()->SetBinLabel(1+par.getParameterIndex(), (parSet.getName() + "/" + par.getTitle()).c_str());

          fitterCovarianceMatrixTH2D->GetXaxis()->SetBinLabel(1+parameterIndexOffset+par.getParameterIndex(), (parSet.getName() + "/" + par.getTitle()).c_str());
          fitterCovarianceMatrixTH2D->GetYaxis()->SetBinLabel(1+parameterIndexOffset+par.getParameterIndex(), (parSet.getName() + "/" + par.getTitle()).c_str());
        }

        GenericToolbox::mkdirTFile(parSetDir, "matrices")->cd();
        covMatrix->Write(Form("Covariance_TMatrixD", parSet.getName().c_str()));
        covMatrixTH2D->Write(Form("Covariance_TH2D", parSet.getName().c_str()));
        corMatrix->Write(Form("Correlation_TMatrixD", parSet.getName().c_str()));
        corMatrixTH2D->Write(Form("Correlation_TH2D", parSet.getName().c_str()));

        // Parameters
        GenericToolbox::mkdirTFile(parSetDir, "values")->cd();
        auto* postFitErrorHist = new TH1D("postFitErrors_TH1D", "Post-fit Errors", parSet.getNbParameters(), 0, parSet.getNbParameters());
        auto* preFitErrorHist = new TH1D("preFitErrors_TH1D", "Pre-fit Errors", parSet.getNbParameters(), 0, parSet.getNbParameters());
        for( const auto& par : parSet.getParameterList() ){
          postFitErrorHist->GetXaxis()->SetBinLabel(1 + par.getParameterIndex(), par.getTitle().c_str());
          postFitErrorHist->SetBinContent( 1 + par.getParameterIndex(), parameterValueList.at( parameterIndexOffset + par.getParameterIndex() ));
          postFitErrorHist->SetBinError( 1 + par.getParameterIndex(), parameterErrorList.at( parameterIndexOffset + par.getParameterIndex() ));

          preFitErrorHist->GetXaxis()->SetBinLabel(1 + par.getParameterIndex(), par.getTitle().c_str());
          preFitErrorHist->SetBinContent( 1 + par.getParameterIndex(), par.getPriorValue() );
          preFitErrorHist->SetBinError( 1 + par.getParameterIndex(), par.getStdDevValue() );
        }

        preFitErrorHist->SetFillColor(kRed-9);
//        preFitErrorHist->SetFillColorAlpha(kRed-9, 0.5);
//        preFitErrorHist->SetFillStyle(4050); // 50 % opaque ?
        preFitErrorHist->SetMarkerStyle(kFullDotLarge);
        preFitErrorHist->SetMarkerColor(kRed-3);
        preFitErrorHist->SetTitle("Pre-fit Errors");

        postFitErrorHist->SetLineColor(9);
        postFitErrorHist->SetLineWidth(2);
        postFitErrorHist->SetMarkerColor(9);
        postFitErrorHist->SetMarkerStyle(kFullDotLarge);
        postFitErrorHist->SetTitle("Post-fit Errors");

        auto* errorsCanvas = new TCanvas("fitConstraints", Form("Fit Constraints for %s", parSet.getName().c_str()), 800, 600);
        errorsCanvas->cd();
        preFitErrorHist->Draw("E2");
        errorsCanvas->Update(); // otherwise does not display...
        postFitErrorHist->Draw("E SAME");

//        auto* legend = gPad->BuildLegend();
        gPad->SetGridx();
        gPad->SetGridy();

        preFitErrorHist->SetTitle(Form("Pre-fit/Post-fit Comparison for %s", parSet.getName().c_str()));
        errorsCanvas->Write("fitConstraints_TCanvas");

        preFitErrorHist->SetTitle(Form("Pre-fit Errors of %s", parSet.getName().c_str()));
        postFitErrorHist->SetTitle(Form("Post-fit Errors of %s", parSet.getName().c_str()));
        postFitErrorHist->Write();
        preFitErrorHist->Write();


        parameterIndexOffset += int(parSet.getNbParameters());
      } // parSet

      errorDir->cd();
      fitterCovarianceMatrix.Write("fitterCovarianceMatrix_TMatrixDSym");
      fitterCovarianceMatrixTH2D->Write("fitterCovarianceMatrix_TH2D");
    } // minimizer valid?
  } // save dir?

}

void FitterEngine::initializePropagator(){

  _propagator_.setConfig(JsonUtils::fetchValue<json>(_config_, "propagatorConfig"));

  TFile* f = TFile::Open(JsonUtils::fetchValue<std::string>(_config_, "mc_file").c_str(), "READ");
  _propagator_.setDataTree( f->Get<TTree>("selectedEvents") );
  _propagator_.setMcFilePath(JsonUtils::fetchValue<std::string>(_config_, "mc_file"));
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
