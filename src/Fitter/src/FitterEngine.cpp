//
// Created by Nadrino on 11/06/2021.
//

#include "FitterEngine.h"

#include "JsonUtils.h"
#include "GlobalVariables.h"

#include "Logger.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.h"

#include <Math/Factory.h>
#include "TGraph.h"
#include "TLegend.h"

#include <cmath>


LoggerInit([]{
  Logger::setUserHeaderStr("[FitterEngine]");
})

FitterEngine::FitterEngine() { this->reset(); }
FitterEngine::~FitterEngine() { this->reset(); }

void FitterEngine::reset() {
  _fitIsDone_ = false;
  _saveDir_ = nullptr;
  _config_.clear();
//  _chi2History_.clear();
  _chi2HistoryTree_ = nullptr;

  _propagator_.reset();
  _minimizer_.reset();
  _functor_.reset();
  _nbFitParameters_ = 0;
  _nbParameters_ = 0;
  _nbFitCalls_ = 0;

  _convergenceMonitor_.reset();
}

void FitterEngine::setSaveDir(TDirectory *saveDir) {
  _saveDir_ = saveDir;
}
void FitterEngine::setConfig(const nlohmann::json &config_) {
  _config_ = config_;
  JsonUtils::forwardConfig(_config_);
}
void FitterEngine::setNbScanSteps(int nbScanSteps) {
  LogThrowIf(nbScanSteps < 0, "Can't provide negative value for _nbScanSteps_")
  _nbScanSteps_ = nbScanSteps;
}
void FitterEngine::setEnablePostFitScan(bool enablePostFitScan) {
  _enablePostFitScan_ = enablePostFitScan;
}


void FitterEngine::initialize() {

  LogThrowIf(_config_.empty(), "Config is not set.");

  _propagator_.setConfig(JsonUtils::fetchValue<json>(_config_, "propagatorConfig"));
  _propagator_.setSaveDir(GenericToolbox::mkdirTFile(_saveDir_, "Propagator"));
  _propagator_.initialize();

  _nbFitParameters_ = 0;
  _nbParameters_ = 0;
  for( const auto& parSet : _propagator_.getParameterSetsList() ){
    _nbParameters_ += int(parSet.getNbParameters());
    if( not parSet.isUseEigenDecompInFit() ){
      _nbFitParameters_ += int(parSet.getNbParameters());
    }
    else{
      _nbFitParameters_ += parSet.getNbEnabledEigenParameters();
    }
  }

  if( JsonUtils::fetchValue(_config_, "scaleParStepWithChi2Response", false) ){
    _parStepGain_ = JsonUtils::fetchValue(_config_, "parStepGain", _parStepGain_);
    LogInfo << "Using parameter step scale: " << _parStepGain_ << std::endl;
    this->rescaleParametersStepSize();
  }

  if( JsonUtils::fetchValue(_config_, "fixGhostFitParameters", false) ) this->fixGhostFitParameters();

  _convergenceMonitor_.addDisplayedQuantity("VarName");
  _convergenceMonitor_.addDisplayedQuantity("LastAddedValue");
  _convergenceMonitor_.addDisplayedQuantity("SlopePerCall");

//  _convergenceMonitor_.getQuantity("VarName").title = "χ² value"; // special chars resize the box
  _convergenceMonitor_.getQuantity("VarName").title = "Likelihood";
  _convergenceMonitor_.getQuantity("LastAddedValue").title = "Current Value";
  _convergenceMonitor_.getQuantity("SlopePerCall").title = "Avg. Slope /call";

  _convergenceMonitor_.addVariable("Total");
  _convergenceMonitor_.addVariable("Stat");
  _convergenceMonitor_.addVariable("Syst");

  if( _saveDir_ != nullptr ){
    GenericToolbox::mkdirTFile(_saveDir_, "fit")->cd();
    _chi2HistoryTree_ = new TTree("chi2History", "chi2History");
    _chi2HistoryTree_->Branch("nbFitCalls", &_nbFitCalls_);
    _chi2HistoryTree_->Branch("chi2Total", &_chi2Buffer_);
    _chi2HistoryTree_->Branch("chi2Stat", &_chi2StatBuffer_);
    _chi2HistoryTree_->Branch("chi2Pulls", &_chi2PullsBuffer_);

    auto* dir = GenericToolbox::mkdirTFile(_saveDir_, "preFit/events");
    _propagator_.getTreeWriter().writeSamples(dir);
  }

  if( JsonUtils::fetchValue(_config_, "throwMcBeforeFit", false) ){
    LogInfo << "Throwing correlated parameters of MC away from their prior..." << std::endl;
    double throwGain = JsonUtils::fetchValue(_config_, "throwMcBeforeFitGain", 1.);
    LogInfo << "Throw gain form MC push set to: " << throwGain << std::endl;

    for( auto& parSet : _propagator_.getParameterSetsList() ){

      if(not parSet.isEnabled()) continue;

      if( not parSet.isEnableThrowMcBeforeFit() ){
        LogWarning << "\"" << parSet.getName() << "\" has marked disabled throwMcBeforeFit: skipping." << std::endl;
        continue;
      }

      if( JsonUtils::doKeyExist(parSet.getConfig(), "customFitParThrow") ){
        LogAlert << "Using custom mc parameter push for " << parSet.getName() << std::endl;
        for(auto& entry : JsonUtils::fetchValue(parSet.getConfig(), "customFitParThrow", std::vector<nlohmann::json>())){
          int parIndex = JsonUtils::fetchValue<int>(entry, "parIndex");

          double pushVal;
          if( parSet.isUseEigenDecompInFit() ){
            pushVal =
                parSet.getEigenParameterValue(parIndex)
                + parSet.getEigenSigma(parIndex)
                  * JsonUtils::fetchValue<double>(entry, "nbSigmaAway");

            LogWarning << "Pushing eigen_#" << parIndex << ": " << parSet.getEigenParameterValue(parIndex) << " → " << pushVal << std::endl;
            parSet.setEigenParameter(parIndex, pushVal);
            parSet.propagateEigenToOriginal();
//          _minimizer_->SetVariableValue(iFitPar + parIndex, parSet.getEigenParameterValue(parIndex) );
          }
          else{
            pushVal =
                parSet.getParameterList()[parIndex].getParameterValue()
                + parSet.getParameterList()[parIndex].getStdDevValue()
                  * JsonUtils::fetchValue<double>(entry, "nbSigmaAway");


            LogWarning << "Pushing #" << parIndex << " to " << pushVal << std::endl;
            parSet.getParameterList()[parIndex].setParameterValue( pushVal );
//          _minimizer_->SetVariableValue(iFitPar + parIndex, parSet.getParameterList()[parIndex].getParameterValue() );
          }

        }
        continue;
      }
      else{
        LogAlert << "Throwing correlated parameters for " << parSet.getName() << std::endl;
        parSet.throwFitParameters(throwGain);
      }

    } // parSet

    _propagator_.preventRfPropagation(); // Making sure since we need the weight of each event
    _propagator_.propagateParametersOnSamples();
  } // throwMcBeforeFit

  this->initializeMinimizer();

}

bool FitterEngine::isFitHasConverged() const {
  return _fitHasConverged_;
}
double FitterEngine::getChi2Buffer() const {
  return _chi2Buffer_;
}
double FitterEngine::getChi2StatBuffer() const {
  return _chi2StatBuffer_;
}
const Propagator FitterEngine::getPropagator() const {
  return _propagator_;
}

void FitterEngine::generateSamplePlots(const std::string& savePath_){
  LogInfo << __METHOD_NAME__ << std::endl;

  _propagator_.preventRfPropagation(); // Making sure since we need the weight of each event
  _propagator_.propagateParametersOnSamples();

  if( not _propagator_.getPlotGenerator().isEmpty() ){
    _propagator_.getPlotGenerator().generateSamplePlots(
        GenericToolbox::mkdirTFile(_saveDir_, savePath_ )
    );
  }
  else{
    LogWarning << "No histogram is defined in the PlotGenerator. Skipping..." << std::endl;
  }

}
void FitterEngine::generateOneSigmaPlots(const std::string& savePath_){

  _propagator_.preventRfPropagation(); // Making sure since we need the weight of each event
  _propagator_.propagateParametersOnSamples();
  _propagator_.getPlotGenerator().generateSamplePlots();

  GenericToolbox::mkdirTFile(_saveDir_, savePath_)->cd();
  auto refHistList = _propagator_.getPlotGenerator().getHistHolderList(); // current buffer

  // +1 sigma
  int iPar = -1;
  for( auto& parSet : _propagator_.getParameterSetsList() ){

    if( not parSet.isEnabled() ) continue;

    if( JsonUtils::fetchValue(parSet.getConfig(), "disableOneSigmaPlots", false) ){
      LogDebug << "+1σ plots disabled for \"" << parSet.getName() << "\"" << std::endl;
      continue;
    }

    for( auto& par : parSet.getParameterList() ){
      iPar++;

      if( not par.isEnabled() ) continue;

      std::string tag;
      if( par.isFixed() ){ tag += "_FIXED"; }

      double currentParValue = par.getParameterValue();
      par.setParameterValue( currentParValue + par.getStdDevValue() );
      LogInfo << "(" << iPar+1 << "/" << _nbParameters_ << ") +1σ on " << parSet.getName() + "/" + par.getTitle()
              << " -> " << par.getParameterValue() << std::endl;
      _propagator_.propagateParametersOnSamples();

      std::string savePath = savePath_;
      if( not savePath.empty() ) savePath += "/";
      savePath += "oneSigma/" + parSet.getName() + "/" + par.getTitle() + tag;
      auto* saveDir = GenericToolbox::mkdirTFile(_saveDir_, savePath );
      saveDir->cd();

      _propagator_.getPlotGenerator().generateSampleHistograms(nullptr, 1);

      auto oneSigmaHistList = _propagator_.getPlotGenerator().getHistHolderList(1);
      _propagator_.getPlotGenerator().generateComparisonPlots( oneSigmaHistList, refHistList, saveDir );
      par.setParameterValue( currentParValue );
      _propagator_.propagateParametersOnSamples();

      const auto& compHistList = _propagator_.getPlotGenerator().getComparisonHistHolderList();

//      // Since those were not saved, delete manually
//      // Don't delete? -> slower each time
////      for( auto& hist : oneSigmaHistList ){ delete hist.histPtr; }
//      oneSigmaHistList.clear();
    }

    if( parSet.isUseEigenDecompInFit() ){
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        double currentParValue = parSet.getEigenParameterValue(iEigen);
        parSet.setEigenParameter(iEigen, currentParValue + parSet.getEigenSigma(iEigen));
        LogInfo << "(" << iEigen+1 << "/" << parSet.getNbEnabledEigenParameters() << ") +1σ on " << parSet.getName() + "/eigen_#" << iEigen
                << " -> " << parSet.getEigenSigma(iEigen) << std::endl;
        parSet.propagateEigenToOriginal();
        _propagator_.propagateParametersOnSamples();

        std::string savePath = savePath_;
        if( not savePath.empty() ) savePath += "/";
        savePath += "oneSigma/" + parSet.getName() + "/eigen_#" + std::to_string(iEigen);
        auto* saveDir = GenericToolbox::mkdirTFile(_saveDir_, savePath );
        saveDir->cd();

        _propagator_.getPlotGenerator().generateSampleHistograms(nullptr, 1);

        auto oneSigmaHistList = _propagator_.getPlotGenerator().getHistHolderList(1);
        _propagator_.getPlotGenerator().generateComparisonPlots( oneSigmaHistList, refHistList, saveDir );
        parSet.setEigenParameter(iEigen, currentParValue);
        parSet.propagateEigenToOriginal();
        _propagator_.propagateParametersOnSamples();

        const auto& compHistList = _propagator_.getPlotGenerator().getComparisonHistHolderList();

        // Since those were not saved, delete manually
//        for( auto& hist : oneSigmaHistList ){ delete hist.histPtr; }
        oneSigmaHistList.clear();
      }
    }

  }

  _saveDir_->cd();

  // Since those were not saved, delete manually
//  for( auto& refHist : refHistList ){ delete refHist.histPtr; }
  refHistList.clear();

}

void FitterEngine::fixGhostFitParameters(){
  LogInfo << __METHOD_NAME__ << std::endl;

  _propagator_.allowRfPropagation(); // since we don't need the weight of each event (only the Chi2 value)
  updateChi2Cache();

  LogDebug << "Reference χ² = " << _chi2StatBuffer_ << std::endl;
  double baseChi2 = _chi2Buffer_;
  double baseChi2Stat = _chi2StatBuffer_;
  double baseChi2Syst = _chi2PullsBuffer_;

  // +1 sigma
  int iFitPar = -1;
  std::stringstream ssPrint;
  double deltaChi2;
  double deltaChi2Stat;
  double deltaChi2Syst;

  for( auto& parSet : _propagator_.getParameterSetsList() ){

    if( not JsonUtils::fetchValue(parSet.getConfig(), "fixGhostFitParameters", false) ) {
      LogWarning << "Skipping \"" << parSet.getName() << "\" as fixGhostFitParameters is set to false" << std::endl;
      if( not parSet.isUseEigenDecompInFit() ) iFitPar += parSet.getNbParameters();
      else iFitPar += parSet.getNbEnabledEigenParameters();
      continue;
    }

    bool fixNextEigenPars{false};
    if( parSet.isUseEigenDecompInFit() ){
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        iFitPar++;
        ssPrint.str("");

        if( fixNextEigenPars ){
          LogInfo << GenericToolbox::ColorCodes::redBackGround << "Fixing next par" << "(" << iFitPar+1 << "/" << _nbFitParameters_
          << ") +1σ on " << parSet.getName() + "/eigen_#" + std::to_string(iEigen) << " -> "
          << parSet.getEigenParameterValue(iEigen) << GenericToolbox::ColorCodes::resetColor << std::endl;
          parSet.setEigenParIsFixed(iEigen, true);
          continue;
        }

        double currentParValue = parSet.getEigenParameterValue(iEigen);
        parSet.setEigenParameter(iEigen, currentParValue + parSet.getEigenSigma(iEigen));
        parSet.propagateEigenToOriginal();

        ssPrint << "(" << iFitPar+1 << "/" << _nbFitParameters_ << ") +1σ on " << parSet.getName() + "/eigen_#" + std::to_string(iEigen)
                << " -> " << parSet.getEigenParameterValue(iEigen);
        LogInfo << ssPrint.str() << "..." << std::endl;

        updateChi2Cache();

        deltaChi2Stat = _chi2StatBuffer_ - baseChi2Stat;
        deltaChi2Syst = _chi2PullsBuffer_ - baseChi2Syst;
        deltaChi2 = _chi2Buffer_ - baseChi2;
        ssPrint << ": Δχ²(stat) = " << deltaChi2Stat << " / Δχ²(syst) = " << deltaChi2Syst;
        ssPrint << ": Δχ² = " << deltaChi2;

        LogInfo.moveTerminalCursorBack(1);
        LogInfo << ssPrint.str() << std::endl;

        if( std::abs(deltaChi2Stat) < JsonUtils::fetchValue(_config_, "ghostParameterDeltaChi2Threshold", 1E-6) ){
          parSet.setEigenParIsFixed(iEigen, true);

          ssPrint << " < " << JsonUtils::fetchValue(_config_, "ghostParameterDeltaChi2Threshold", 1E-6) << " -> " << "FIXED";
          LogInfo.moveTerminalCursorBack(1);
          LogInfo << GenericToolbox::ColorCodes::redBackGround << ssPrint.str() << GenericToolbox::ColorCodes::resetColor << std::endl;

          if( JsonUtils::fetchValue(_config_, "fixGhostEigenParmetersAfterFirstRejected", false) ) fixNextEigenPars = true;
        }

        parSet.setEigenParameter(iEigen, currentParValue);
        parSet.propagateEigenToOriginal();
      }
    }
    else{
      for( auto& par : parSet.getParameterList() ){
        iFitPar++;
        ssPrint.str("");

        if( par.isEnabled() and not par.isFixed() ){
          double currentParValue = par.getParameterValue();
          par.setParameterValue( currentParValue + par.getStdDevValue() );

          ssPrint << "(" << iFitPar+1 << "/" << _nbFitParameters_ << ") +1σ on " << parSet.getName() + "/" + par.getTitle()
                  << " " << currentParValue << " -> " << par.getParameterValue();
          LogInfo << ssPrint.str() << "..." << std::endl;

          updateChi2Cache();
          deltaChi2Stat = _chi2StatBuffer_ - baseChi2Stat;
          deltaChi2Syst = _chi2PullsBuffer_ - baseChi2Syst;
          deltaChi2 = _chi2Buffer_ - baseChi2;
          ssPrint << ": Δχ²(stat) = " << deltaChi2Stat << " / Δχ²(syst) = " << deltaChi2Syst;
          ssPrint << ": Δχ² = " << deltaChi2;

          LogInfo.moveTerminalCursorBack(1);
          LogInfo << ssPrint.str() << std::endl;

          if( std::abs(deltaChi2Stat) < JsonUtils::fetchValue(_config_, "ghostParameterDeltaChi2Threshold", 1E-6) ){
            par.setIsFixed(true); // ignored in the Chi2 computation of the parSet
            ssPrint << " < " << JsonUtils::fetchValue(_config_, "ghostParameterDeltaChi2Threshold", 1E-6) << " -> " << "FIXED";
            LogInfo.moveTerminalCursorBack(1);
            LogInfo << GenericToolbox::ColorCodes::redBackGround << ssPrint.str() << GenericToolbox::ColorCodes::resetColor << std::endl;
          }

          par.setParameterValue( currentParValue );
        }

      }
    }

  }

  updateChi2Cache(); // comeback to old values
  _propagator_.preventRfPropagation();
}
void FitterEngine::scanParameters(int nbSteps_, const std::string &saveDir_) {
  LogInfo << "Performing parameter scans..." << std::endl;
  for( int iPar = 0 ; iPar < _minimizer_->NDim() ; iPar++ ){
    if( _minimizer_->IsFixedVariable(iPar) ) continue;
    this->scanParameter(iPar, nbSteps_, saveDir_);
  } // iPar
}
void FitterEngine::scanParameter(int iPar, int nbSteps_, const std::string &saveDir_) {

  if( nbSteps_ < 0 ){ nbSteps_ = _nbScanSteps_; }

//  double originalParValue = fetchCurrentParameterValue(iPar);

  //Internally Scan performs steps-1, so add one to actually get the number of steps
  //we ask for.
  unsigned int adj_steps = nbSteps_+1;
  auto* x = new double[adj_steps] {};
  auto* y = new double[adj_steps] {};

  LogInfo << "Scanning fit parameter #" << iPar
          << ": " << _minimizer_->VariableName(iPar) << " / " << nbSteps_ << " steps..." << std::endl;

  _propagator_.allowRfPropagation();
  bool success = _minimizer_->Scan(iPar, adj_steps, x, y);

  if( not success ){
    LogError << "Parameter scan failed." << std::endl;
  }

  TGraph scanGraph(nbSteps_, x, y);

  std::stringstream ss;
  ss << GenericToolbox::replaceSubstringInString(_minimizer_->VariableName(iPar), "/", "_");
  ss << "_TGraph";

  scanGraph.SetTitle(_fitIsDone_ ? "Post-fit scan": "Pre-fit scan");
  scanGraph.GetYaxis()->SetTitle("LLH");
  scanGraph.GetYaxis()->SetTitle(_minimizer_->VariableName(iPar).c_str());

  if( _saveDir_ != nullptr ){
    GenericToolbox::mkdirTFile(_saveDir_, saveDir_)->cd();
    scanGraph.Write( ss.str().c_str() );
  }
  _propagator_.preventRfPropagation();

//  _minimizer_->SetVariableValue(iPar, originalParValue);
//  this->updateParameterValue(iPar, originalParValue);
//  updateChi2Cache();

  delete[] x;
  delete[] y;
}

void FitterEngine::fit(){
  LogWarning << __METHOD_NAME__ << std::endl;

  LogWarning << std::endl << GenericToolbox::addUpDownBars("Summary of the fit parameters:") << std::endl;
  int iFitPar = -1;
  for( const auto& parSet : _propagator_.getParameterSetsList() ){
    if( not parSet.isUseEigenDecompInFit() ){
      LogWarning << parSet.getName() << ": " << parSet.getNbParameters() << " parameters" << std::endl;
      Logger::setIndentStr("├─ ");
      for( const auto& par : parSet.getParameterList() ){
        iFitPar++;
        if( par.isEnabled() ){
          if( _minimizer_->IsFixedVariable(iFitPar) ){
            LogInfo << "\033[41m" << "#" << iFitPar << " -> " << parSet.getName() << "/" << par.getTitle() << ": FIXED - Prior: " << par.getParameterValue() << " ± " << par.getStdDevValue() <<  "\033[0m" << std::endl;
          }
          else{
            LogInfo << "#" << iFitPar << " -> " << parSet.getName() << "/" << par.getTitle() << " - Prior: " << par.getParameterValue() << " ± " << par.getStdDevValue() << std::endl;
          }
        }
        else{
          LogInfo << "\033[43m" << "#" << iFitPar << " -> " << parSet.getName() << "/" << par.getTitle() << ": Disabled" <<  "\033[0m" << std::endl;
        }
      }
      Logger::setIndentStr("");
    }
    else{
      LogWarning << parSet.getName() << ": " << parSet.getNbEnabledEigenParameters() << " eigen parameters" << std::endl;
      Logger::setIndentStr("├─ ");
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        iFitPar++;
        if( _minimizer_->IsFixedVariable(iFitPar) ) {
          LogInfo << "\033[41m" << "#" << iFitPar << " -> " << parSet.getName() << "/eigen_#" << iEigen << ": FIXED - Prior: "
                  << parSet.getEigenParameterValue(iEigen) << " ± " << parSet.getEigenSigma(iEigen) << "\033[0m" << std::endl;
        }
        else{
          LogInfo << "#" << iFitPar << " -> " << parSet.getName() << "/eigen_#" << iEigen << " - Prior: "
                  << parSet.getEigenParameterValue(iEigen) << " ± " << parSet.getEigenSigma(iEigen) << std::endl;
        }
      }
      Logger::setIndentStr("");
    }
  }

  _propagator_.allowRfPropagation(); // if RF are setup -> a lot faster
  updateChi2Cache();

  LogWarning << std::endl << GenericToolbox::addUpDownBars("Calling minimize...") << std::endl;
  LogInfo << "Number of defined parameters: " << _minimizer_->NDim() << std::endl
          << "Number of free parameters   : " << _minimizer_->NFree() << std::endl
          << "Number of fixed parameters  : " << _minimizer_->NDim() - _minimizer_->NFree()
          << std::endl;

  int nbFitCallOffset = _nbFitCalls_;
  LogInfo << "Fit call offset: " << nbFitCallOffset << std::endl;
  _enableFitMonitor_ = true;
  _fitHasConverged_ = _minimizer_->Minimize();
  _enableFitMonitor_ = false;
  int nbMinimizeCalls = _nbFitCalls_ - nbFitCallOffset;

  LogInfo << _convergenceMonitor_.generateMonitorString(); // lasting printout
  LogInfo << "Minimization ended after " << nbMinimizeCalls << " calls." << std::endl;
  if(_minimizerAlgo_ == "Migrad") LogWarning << "Status code: " << minuitStatusCodeStr.at(_minimizer_->Status()) << std::endl;
  else LogWarning << "Status code: " << _minimizer_->Status() << std::endl;
  if(_minimizerAlgo_ == "Migrad") LogWarning << "Covariance matrix status code: " << covMatrixStatusCodeStr.at(_minimizer_->CovMatrixStatus()) << std::endl;
  else LogWarning << "Covariance matrix status code: " << _minimizer_->CovMatrixStatus() << std::endl;
  if( _saveDir_ != nullptr ){
    GenericToolbox::mkdirTFile(_saveDir_, "fit")->cd();
    _chi2HistoryTree_->Write();
  }

  if( _fitHasConverged_ ){
    LogInfo << "Fit converged!" << std::endl;
    LogInfo << _convergenceMonitor_.generateMonitorString(); // lasting printout
  }
  else{
    LogError << "Did not converged." << std::endl;
    LogError << _convergenceMonitor_.generateMonitorString(); // lasting printout
  }

  LogInfo << "Writing " << _minimizerType_ << "/" << _minimizerAlgo_ << " post-fit errors" << std::endl;
  this->writePostFitData(GenericToolbox::mkdirTFile(_saveDir_, "postFit/" + _minimizerAlgo_));

  if( _enablePostFitScan_ ){
    LogInfo << "Scanning parameters around the minimum point..." << std::endl;
    this->scanParameters(-1, "postFit/scan");
  }

  if( _fitHasConverged_ ){
    LogInfo << "Evaluating post-fit errors..." << std::endl;

    _enableFitMonitor_ = true;
    if( JsonUtils::fetchValue(_minimizerConfig_, "enablePostFitErrorFit", true) ){
      std::string errorAlgo = JsonUtils::fetchValue(_minimizerConfig_, "errors", "Hesse");
      if     ( errorAlgo == "Minos" ){

        // Put back at minimum
        iFitPar = -1;
        LogInfo << "Releasing parameters for error evaluation..." << std::endl;
        for( auto& parSet : _propagator_.getParameterSetsList() ){

          if( not JsonUtils::fetchValue(parSet.getConfig(), "releaseFixedParametersOnHesse", true) ){
            continue;
          }

          if( not parSet.isUseEigenDecompInFit() ){
            for( auto& par : parSet.getParameterList() ){
              if( par.isFixed() ){
                LogDebug << "Releasing " << parSet.getName() << "/" << par.getTitle() << std::endl;
                par.setIsFixed(false);
                _minimizer_->ReleaseVariable(iFitPar++);
              }

            }
          }
          else{
            for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
              LogDebug << "Releasing " << parSet.getName() << "/eigen_#" << iEigen << std::endl;
              _minimizer_->ReleaseVariable(iFitPar++);
              parSet.setEigenParIsFixed(iEigen, false);
            }
          }

        }

        LogWarning << std::endl << GenericToolbox::addUpDownBars("Calling MINOS...") << std::endl;

        double errLow, errHigh;
        _minimizer_->SetPrintLevel(0);

        iFitPar = -1;
        for( auto& parSet : _propagator_.getParameterSetsList() ){
          if( JsonUtils::fetchValue(parSet.getConfig(), "skipMinos", false) ){
            LogWarning << "Minos error evaluation is disabled for parSet: " << parSet.getName() << std::endl;
            continue;
          }

          if( not parSet.isUseEigenDecompInFit() ){
            for( auto& par : parSet.getParameterList() ){
              iFitPar++;
              if( _minimizer_->IsFixedVariable(iFitPar) ) continue;

              LogInfo << "Evaluating: " << _minimizer_->VariableName(iFitPar) << "..." << std::endl;
              bool isOk = _minimizer_->GetMinosError(iFitPar, errLow, errHigh);
#if ROOT_VERSION_CODE >= ROOT_VERSION(6,23,02)
              LogWarning << minosStatusCodeStr.at(_minimizer_->MinosStatus()) << std::endl;
#endif
              if( isOk ){
                LogInfo << _minimizer_->VariableName(iFitPar) << ": " << errLow << " <- " << _minimizer_->X()[iFitPar] << " -> +" << errHigh << std::endl;
              }
              else{
                LogError << _minimizer_->VariableName(iFitPar) << ": " << errLow << " <- " << _minimizer_->X()[iFitPar] << " -> +" << errHigh
                         << " - MINOS returned an error." << std::endl;
              }
            }
          }
          else{
            for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
              iFitPar++;
              if( _minimizer_->IsFixedVariable(iFitPar) ) continue;

              LogInfo << "Evaluating: " << _minimizer_->VariableName(iFitPar) << "..." << std::endl;
              bool isOk = _minimizer_->GetMinosError(iFitPar, errLow, errHigh);
#if ROOT_VERSION_CODE >= ROOT_VERSION(6,23,02)
              LogWarning << minosStatusCodeStr.at(_minimizer_->MinosStatus()) << std::endl;
#endif
              if( isOk ){
                LogInfo << _minimizer_->VariableName(iFitPar) << ": " << errLow << " <- " << _minimizer_->X()[iFitPar] << " -> +" << errHigh << std::endl;
              }
              else{
                LogError << _minimizer_->VariableName(iFitPar) << ": " << errLow << " <- " << _minimizer_->X()[iFitPar] << " -> +" << errHigh
                         << " - MINOS returned an error." << std::endl;
              }
            }
          }
        }

        // Put back at minimum
        iFitPar = -1;
        for( auto& parSet : _propagator_.getParameterSetsList() ){

          if( not parSet.isUseEigenDecompInFit() ){
            for( auto& par : parSet.getParameterList() ){
              iFitPar++;
              par.setParameterValue(_minimizer_->X()[iFitPar]);
            }
          }
          else{
            for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
              iFitPar++;
              parSet.setEigenParameter(iEigen, _minimizer_->X()[iFitPar]);
            }
            parSet.propagateEigenToOriginal();
          }

        }
        updateChi2Cache();
      } // Minos
      else if( errorAlgo == "Hesse" ){

        if( JsonUtils::fetchValue(_config_, "restoreStepSizeBeforeHesse", false) ){
          LogWarning << "Restoring step size before HESSE..." << std::endl;
          int iPar = -1;
          for( auto& parSet : _propagator_.getParameterSetsList() ){

            if( not parSet.isUseEigenDecompInFit() ){
              for( auto& par : parSet.getParameterList()  ){
                iPar++;
                if(not _useNormalizedFitSpace_){ _minimizer_->SetVariableStepSize(iPar, par.getStepSize()); }
                else{ _minimizer_->SetVariableStepSize(iPar, FitParameterSet::toNormalizedParRange(par.getStepSize(), par)); } // should be 1
              } // par
            }
            else{
              for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
                iPar++;
                if(not _useNormalizedFitSpace_){ _minimizer_->SetVariableStepSize(iPar, parSet.getEigenParStepSize(iEigen)); }
                else{ _minimizer_->SetVariableStepSize(iPar, parSet.toNormalizedEigenParRange(parSet.getEigenParStepSize(iEigen), iEigen)); } // should be 1
              }
            }

          } // parSet
        }

        LogWarning << std::endl << GenericToolbox::addUpDownBars("Calling HESSE...") << std::endl;
        LogInfo << "Number of defined parameters: " << _minimizer_->NDim() << std::endl
                << "Number of free parameters   : " << _minimizer_->NFree() << std::endl
                << "Number of fixed parameters  : " << _minimizer_->NDim() - _minimizer_->NFree()
                << std::endl;

        nbFitCallOffset = _nbFitCalls_;
        LogInfo << "Fit call offset: " << nbFitCallOffset << std::endl;

        _fitHasConverged_ = _minimizer_->Hesse();
        LogInfo << "Hesse ended after " << _nbFitCalls_ - nbFitCallOffset << " calls." << std::endl;
        LogWarning << "HESSE status code: " << hesseStatusCodeStr.at(_minimizer_->Status()) << std::endl;
        LogWarning << "Covariance matrix status code: " << covMatrixStatusCodeStr.at(_minimizer_->CovMatrixStatus()) << std::endl;

        if( _minimizer_->CovMatrixStatus() == 2 ){ _isBadCovMat_ = true; }

        if(not _fitHasConverged_){
          LogError  << "Hesse did not converge." << std::endl;
          LogError << _convergenceMonitor_.generateMonitorString(); // lasting printout
        }
        else{
          LogInfo << "Hesse converged." << std::endl;
          LogInfo << _convergenceMonitor_.generateMonitorString(); // lasting printout
        }

        LogInfo << "Writing HESSE post-fit errors" << std::endl;
        this->writePostFitData(GenericToolbox::mkdirTFile(_saveDir_, "postFit/Hesse"));
      }
      else{
        LogError << GET_VAR_NAME_VALUE(errorAlgo) << " not implemented." << std::endl;
      }
    }
    _enableFitMonitor_ = false;
  }

  _propagator_.preventRfPropagation(); // since we need the weight of each event
  _propagator_.propagateParametersOnSamples();

  _fitIsDone_ = true;
}
void FitterEngine::updateChi2Cache(){

  double buffer;

  // Propagate on histograms
  _propagator_.propagateParametersOnSamples();

  ////////////////////////////////
  // Compute chi2 stat
  ////////////////////////////////
  _chi2StatBuffer_ = _propagator_.getFitSampleSet().evalLikelihood();

  ////////////////////////////////
  // Compute the penalty terms
  ////////////////////////////////
  _chi2PullsBuffer_ = 0;
  _chi2RegBuffer_ = 0;
  for( auto& parSet : _propagator_.getParameterSetsList() ){
    buffer = parSet.getChi2();
    _chi2PullsBuffer_ += buffer;
  }

  _chi2Buffer_ = _chi2StatBuffer_ + _chi2PullsBuffer_ + _chi2RegBuffer_;

}
double FitterEngine::evalFit(const double* parArray_){
  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  if(_nbFitCalls_ != 0){
    _outEvalFitAvgTimer_.counts++ ; _outEvalFitAvgTimer_.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("out_evalFit");
  }
  _nbFitCalls_++;

  // Update fit parameter values:
  int iFitPar = -1;
  for( auto& parSet : _propagator_.getParameterSetsList() ){
    if( not parSet.isUseEigenDecompInFit() ){
      for( auto& par : parSet.getParameterList() ){
        iFitPar++;
        if( not _useNormalizedFitSpace_ ){
          par.setParameterValue( parArray_[iFitPar] );
        }
        else{
          par.setParameterValue( FitParameterSet::toRealParValue(parArray_[iFitPar], par) );
        }
      }
    }
    else{
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        iFitPar++;
        if( not _useNormalizedFitSpace_ ){
          parSet.setEigenParameter(iEigen, parArray_[iFitPar]);
        }
        else{
          parSet.setEigenParameter(iEigen, parSet.toRealEigenParValue(parArray_[iFitPar], iEigen));
        }
      }
      parSet.propagateEigenToOriginal();
    }
  }

  // Compute the Chi2
  updateChi2Cache();

  _evalFitAvgTimer_.counts++; _evalFitAvgTimer_.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);

  if(_convergenceMonitor_.isGenerateMonitorStringOk() and _enableFitMonitor_ ){

    if( _itSpeed_.counts != 0 ){
      _itSpeed_.counts = _nbFitCalls_ - _itSpeed_.counts; // how many cycles since last print
      _itSpeed_.cumulated = GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("itSpeed"); // time since last print
    }
    else{
      _itSpeed_.counts = _nbFitCalls_;
      GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("itSpeed");
    }


    std::stringstream ss;
    ss << __METHOD_NAME__ << ": call #" << _nbFitCalls_;
    ss << std::endl << "Current RAM: " << GenericToolbox::parseSizeUnits(GenericToolbox::getProcessMemoryUsage());
    ss << std::endl << "Avg χ² computation time: " << _evalFitAvgTimer_;
    if( not _propagator_.isUseResponseFunctions() ){
      ss << std::endl << "├─ Current speed: " << (double)_itSpeed_.counts/(double)_itSpeed_.cumulated * 1E6 << " it/s";
      ss << std::endl << "├─ Avg time for " << _minimizerType_ << "/" << _minimizerAlgo_ << ": " << _outEvalFitAvgTimer_;
      ss << std::endl << "├─ Avg time to propagate weights: " << _propagator_.weightProp;
      ss << std::endl << "├─ Avg time to fill histograms: " << _propagator_.fillProp;
    }
    else{
      ss << GET_VAR_NAME_VALUE(_propagator_.applyRf);
    }
    _convergenceMonitor_.setHeaderString(ss.str());
    _convergenceMonitor_.getVariable("Total").addQuantity(_chi2Buffer_);
    _convergenceMonitor_.getVariable("Stat").addQuantity(_chi2StatBuffer_);
    _convergenceMonitor_.getVariable("Syst").addQuantity(_chi2PullsBuffer_);

    if( _nbFitCalls_ == 1 ){
      LogInfo << _convergenceMonitor_.generateMonitorString();
    }
    else{
      LogInfo << _convergenceMonitor_.generateMonitorString(true);
    }

    _itSpeed_.counts = _nbFitCalls_;
  }

  // Fill History
  _chi2HistoryTree_->Fill();
//  _chi2History_["Total"].emplace_back(_chi2Buffer_);
//  _chi2History_["Stat"].emplace_back(_chi2StatBuffer_);
//  _chi2History_["Syst"].emplace_back(_chi2PullsBuffer_);

  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("out_evalFit");
  return _chi2Buffer_;
}

void FitterEngine::writePostFitData(TDirectory* saveDir_) {
  LogInfo << __METHOD_NAME__ << std::endl;

  LogThrowIf(saveDir_==nullptr, "Save dir not specified")

  this->generateSamplePlots("postFit/samples");

  LogInfo << "Extracting post-fit covariance matrix" << std::endl;
  auto* matricesDir = GenericToolbox::mkdirTFile(saveDir_, "matrices");

  TMatrixDSym totalCovMatrix(int(_minimizer_->NDim()));
  _minimizer_->GetCovMatrix(totalCovMatrix.GetMatrixArray());

  std::function<void(TDirectory*)> decomposeCovarianceMatrixFct = [&](TDirectory* outDir_){
    GenericToolbox::writeInTFile(outDir_, BIND_VAR_REF_NAME(totalCovMatrix));
    TH2D* totalCovTH2D = GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) &totalCovMatrix);
    GenericToolbox::writeInTFile(outDir_, totalCovTH2D, "totalCovMatrix");

    LogInfo << "Eigen decomposition of the post-fit covariance matrix" << std::endl;
    TMatrixDSymEigen decompFitterCovarianceMatrix(totalCovMatrix);
    GenericToolbox::writeInTFile(outDir_, &decompFitterCovarianceMatrix.GetEigenVectors(), "totalCovEigenVectors");
    GenericToolbox::writeInTFile(outDir_,
                                 GenericToolbox::convertTMatrixDtoTH2D(&decompFitterCovarianceMatrix.GetEigenVectors()),
                                 "totalCovEigenVectors");
    GenericToolbox::writeInTFile(outDir_, &decompFitterCovarianceMatrix.GetEigenValues(), "totalCovEigenValues");
    GenericToolbox::writeInTFile(outDir_,
                                 GenericToolbox::convertTVectorDtoTH1D(&decompFitterCovarianceMatrix.GetEigenValues()),
                                 "totalCovEigenValues");

    double conditioning = decompFitterCovarianceMatrix.GetEigenValues().Min()/decompFitterCovarianceMatrix.GetEigenValues().Max();
    LogWarning << "Post-fit error conditioning is: " << conditioning << std::endl;

    if(true){
      LogInfo << "Eigen breakdown..." << std::endl;
      TH1D eigenBreakdownHist("eigenBreakdownHist", "eigenBreakdownHist",
                              int(_minimizer_->NDim()), -0.5, int(_minimizer_->NDim()) - 0.5);
      std::vector<TH1D> eigenBreakdownAccum(decompFitterCovarianceMatrix.GetEigenValues().GetNrows(), eigenBreakdownHist);
      TH1D* lastAccumHist{nullptr};
      std::string progressTitle = LogWarning.getPrefixString() + "Accumulating eigen components...";
      for (int iEigen = decompFitterCovarianceMatrix.GetEigenValues().GetNrows()-1; iEigen >= 0; iEigen--) {
        GenericToolbox::displayProgressBar(decompFitterCovarianceMatrix.GetEigenValues().GetNrows()-iEigen, decompFitterCovarianceMatrix.GetEigenValues().GetNrows(), progressTitle);
        // iEigen = 0 -> biggest error contribution
        // Drawing in the back -> iEigen = 0 should be last in the accum plot
        if( lastAccumHist != nullptr ) eigenBreakdownAccum[iEigen] = *lastAccumHist;
        else eigenBreakdownAccum[iEigen] = eigenBreakdownHist;
        lastAccumHist = &eigenBreakdownAccum[iEigen];

        eigenBreakdownHist.SetTitle(Form("Parameter breakdown for eigen #%i = %f", iEigen,
                                         decompFitterCovarianceMatrix.GetEigenValues()[iEigen]));
        eigenBreakdownHist.SetLineColor(GenericToolbox::defaultColorWheel[iEigen%int(GenericToolbox::defaultColorWheel.size())]);
        eigenBreakdownHist.SetLabelSize(0.02);
        for (int iPar = int(_minimizer_->NDim())-1; iPar >= 0; iPar--) {
          eigenBreakdownHist.SetBinContent(iPar + 1,
                                           decompFitterCovarianceMatrix.GetEigenVectors()[iPar][iEigen]*
                                           decompFitterCovarianceMatrix.GetEigenVectors()[iPar][iEigen]*
                                           decompFitterCovarianceMatrix.GetEigenValues()[iEigen]
          );
          eigenBreakdownHist.GetXaxis()->SetBinLabel(iPar + 1, _minimizer_->VariableName(iPar).c_str());
          eigenBreakdownAccum[iEigen].GetXaxis()->SetBinLabel(iPar + 1, _minimizer_->VariableName(iPar).c_str());
        }
        GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(outDir_, "eigenBreakdown"), &eigenBreakdownHist,
                                     Form("eigen#%i", iEigen));

        eigenBreakdownAccum[iEigen].Add(&eigenBreakdownHist);
        eigenBreakdownAccum[iEigen].SetLabelSize(0.02);
        eigenBreakdownAccum[iEigen].SetLineColor(kBlack);
        eigenBreakdownAccum[iEigen].SetFillColor(GenericToolbox::defaultColorWheel[iEigen%int(GenericToolbox::defaultColorWheel.size())]);

        int cycle = iEigen/int(GenericToolbox::defaultColorWheel.size());
        if( cycle > 0 ) eigenBreakdownAccum[iEigen].SetFillStyle( 3044 + 100 * (cycle%10) );
        else eigenBreakdownAccum[iEigen].SetFillStyle(1001);
      }

      TCanvas accumPlot("accumPlot", "accumPlot", 1280, 720);
      TLegend l(0.15, 0.4, 0.3, 0.85);
      bool isFirst{true};
      for (int iEigen = 0; iEigen < int(eigenBreakdownAccum.size()); iEigen++) {
        if( iEigen < GenericToolbox::defaultColorWheel.size() ){
          l.AddEntry(&eigenBreakdownAccum[iEigen], Form("Eigen #%i = %f", iEigen, decompFitterCovarianceMatrix.GetEigenValues()[iEigen]));
        }
        accumPlot.cd();
        if( isFirst ){
          eigenBreakdownAccum[iEigen].SetTitle("Hessian eigen composition of post-fit errors");
          eigenBreakdownAccum[iEigen].GetYaxis()->SetTitle("Post-fit #sigma^{2}");
          eigenBreakdownAccum[iEigen].Draw("HIST");
        }
        else{
          eigenBreakdownAccum[iEigen].Draw("HIST SAME");
        }
        isFirst = false;
      }
      l.Draw();
      gPad->SetGridx();
      gPad->SetGridy();
      GenericToolbox::writeInTFile(outDir_, &accumPlot, "eigenBreakdown");
    }


    if(true){
      LogInfo << "Parameters breakdown..." << std::endl;
      TH1D parBreakdownHist("parBreakdownHist", "parBreakdownHist",
                            decompFitterCovarianceMatrix.GetEigenValues().GetNrows(), -0.5,
                            decompFitterCovarianceMatrix.GetEigenValues().GetNrows()-0.5);
      std::vector<TH1D> parBreakdownAccum(_minimizer_->NDim());
      TH1D* lastAccumHist{nullptr};
      for (int iPar = int(_minimizer_->NDim())-1; iPar >= 0; iPar--){

        if( lastAccumHist != nullptr ) parBreakdownAccum[iPar] = *lastAccumHist;
        else parBreakdownAccum[iPar] = parBreakdownHist;
        lastAccumHist = &parBreakdownAccum[iPar];

        parBreakdownHist.SetLineColor(GenericToolbox::defaultColorWheel[iPar%int(GenericToolbox::defaultColorWheel.size())]);

        parBreakdownHist.SetTitle(Form("Eigen breakdown for parameter #%i: %s", iPar, _minimizer_->VariableName(iPar).c_str()));
        for (int iEigen = decompFitterCovarianceMatrix.GetEigenValues().GetNrows()-1; iEigen >= 0; iEigen--){
          parBreakdownHist.SetBinContent(
              iPar+1,
              decompFitterCovarianceMatrix.GetEigenVectors()[iPar][iEigen]
              *decompFitterCovarianceMatrix.GetEigenVectors()[iPar][iEigen]
              *decompFitterCovarianceMatrix.GetEigenValues()[iEigen]
          );
        }
        GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(outDir_, "parBreakdown"), &parBreakdownHist,
                                     Form("par#%i", iPar));

        parBreakdownAccum[iPar].Add(&parBreakdownHist);
        parBreakdownAccum[iPar].SetLabelSize(0.02);
        parBreakdownAccum[iPar].SetLineColor(kBlack);
        parBreakdownAccum[iPar].SetFillColor(GenericToolbox::defaultColorWheel[iPar%int(GenericToolbox::defaultColorWheel.size())]);
      }
      TCanvas accumPlot("accumParPlot", "accumParPlot", 1280, 720);
      bool isFirst{true};
      for (int iPar = 0; iPar < int(parBreakdownAccum.size()); iPar++) {
        accumPlot.cd();
        isFirst? parBreakdownAccum[iPar].Draw("HIST"): parBreakdownAccum[iPar].Draw("HIST SAME");
        isFirst = false;
      }
      GenericToolbox::writeInTFile(outDir_, &accumPlot, "parBreakdown");
    }


    auto eigenValuesInv = TVectorD(decompFitterCovarianceMatrix.GetEigenValues());
    for( int iEigen = 0 ; iEigen < eigenValuesInv.GetNrows() ; iEigen++ ){ eigenValuesInv[iEigen] = 1./eigenValuesInv[iEigen]; }
    auto& diagonalMatrixInv = *GenericToolbox::makeDiagonalMatrix(&eigenValuesInv);
    auto invEigVectors = TMatrixD(decompFitterCovarianceMatrix.GetEigenVectors());
    invEigVectors.T();

    LogInfo << "Reconstructing hessian matrix" << std::endl;
    TMatrixD hessianMatrix(int(_minimizer_->NDim()), int(_minimizer_->NDim())); hessianMatrix.Zero();
    hessianMatrix += decompFitterCovarianceMatrix.GetEigenVectors();
    hessianMatrix *= diagonalMatrixInv;
    hessianMatrix *= invEigVectors;
    GenericToolbox::writeInTFile(outDir_, BIND_VAR_REF_NAME(hessianMatrix));
    GenericToolbox::writeInTFile(outDir_, GenericToolbox::convertTMatrixDtoTH2D(&hessianMatrix), "hessianMatrix");
  };

  if( _useNormalizedFitSpace_ ){
    LogInfo << "Writing normalized decomposition of the output matrix..." << std::endl;
    decomposeCovarianceMatrixFct(GenericToolbox::mkdirTFile(matricesDir, "normalizedFitSpace"));

    auto* totalCorrelationMatrix = GenericToolbox::convertToCorrelationMatrix((TMatrixD*) &totalCovMatrix);

    // Convert the diagonal
    int parameterIndexOffset = 0;
    for( const auto& parSet : _propagator_.getParameterSetsList() ){
      if( not parSet.isUseEigenDecompInFit() ){
        int iPar = -1;
        for( const auto& par : parSet.getParameterList() ){
          iPar++;
          double normedError = TMath::Sqrt(totalCovMatrix[parameterIndexOffset + iPar][parameterIndexOffset + iPar]);
          double realError = FitParameterSet::toRealParRange(normedError, par);
          totalCovMatrix[parameterIndexOffset + iPar][parameterIndexOffset + iPar] = realError*realError;
        }
        parameterIndexOffset += int(parSet.getNbParameters());
      }
      else{
        for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
          double normedError = TMath::Sqrt(totalCovMatrix[parameterIndexOffset + iEigen][parameterIndexOffset + iEigen]);
          double realError = parSet.toRealEigenParRange(normedError, iEigen);
          totalCovMatrix[parameterIndexOffset + iEigen][parameterIndexOffset + iEigen] = realError*realError;
        }
        parameterIndexOffset += parSet.getNbEnabledEigenParameters();
      }
    }

    // Convert off-diagonal terms
    for( int iRow = 0 ; iRow < totalCovMatrix.GetNrows() ; iRow++ ){
      for( int iCol = 0 ; iCol < totalCovMatrix.GetNcols() ; iCol++ ){
        if( iRow == iCol ) continue;
        totalCovMatrix[iRow][iCol] = (*totalCorrelationMatrix)[iRow][iCol] * TMath::Sqrt(totalCovMatrix[iRow][iRow]) * TMath::Sqrt(totalCovMatrix[iCol][iCol]);
      }
    }
  }

  LogInfo << "Writing decomposition of the output matrix..." << std::endl;
  decomposeCovarianceMatrixFct(matricesDir);


  TH2D* totalCovTH2D = GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) &totalCovMatrix);


  LogInfo << "Fitter covariance matrix is " << totalCovMatrix.GetNrows() << "x" << totalCovMatrix.GetNcols() << std::endl;
  auto* errorDir = GenericToolbox::mkdirTFile(saveDir_, "errors");

  int parameterIndexOffset = 0;
  for( const auto& parSet : _propagator_.getParameterSetsList() ){

    if( not parSet.isEnabled() ){ continue; }

    LogInfo << "Extracting post-fit errors of parameter set: " << parSet.getName() << std::endl;
    auto* parSetDir = GenericToolbox::mkdirTFile(errorDir, parSet.getName());

    TMatrixD* covMatrix;
    if( not parSet.isUseEigenDecompInFit() ) {
      covMatrix = new TMatrixD(int(parSet.getParameterList().size()), int(parSet.getParameterList().size()));
      for (const auto &parRow: parSet.getParameterList()) {
        for (const auto &parCol: parSet.getParameterList()) {
          (*covMatrix)[parRow.getParameterIndex()][parCol.getParameterIndex()] =
              totalCovMatrix[parameterIndexOffset + parRow.getParameterIndex()][parameterIndexOffset +
                                                                                parCol.getParameterIndex()];
        } // par Y
      } // par X

      for( const auto& par : parSet.getParameterList() ) {
        totalCovTH2D->GetXaxis()->SetBinLabel(1 + parameterIndexOffset + par.getParameterIndex(),
                                              (parSet.getName() + "/" + par.getTitle()).c_str());
        totalCovTH2D->GetYaxis()->SetBinLabel(1 + parameterIndexOffset + par.getParameterIndex(),
                                              (parSet.getName() + "/" + par.getTitle()).c_str());
      }
    }
    else{

      covMatrix = new TMatrixD(parSet.getNbParameters(), parSet.getNbParameters());
      covMatrix->Zero();
      for( int iEigen = 0 ; iEigen < parSet.getNbParameters() ; iEigen++ ){
        for( int jEigen = 0 ; jEigen < parSet.getNbParameters() ; jEigen++ ){
          if(    iEigen >= parSet.getNbEnabledEigenParameters()
              or parSet.isEigenParFixed(iEigen)
              or jEigen >= parSet.getNbEnabledEigenParameters()
              or parSet.isEigenParFixed(jEigen)
              ){
            (iEigen == jEigen) ? (*covMatrix)[iEigen][jEigen] = parSet.getEigenValue(iEigen) : (*covMatrix)[iEigen][jEigen] = 0;
          }
          else{
            (*covMatrix)[iEigen][jEigen] = totalCovMatrix[parameterIndexOffset + iEigen][parameterIndexOffset + jEigen];
          }

        }
      }

      auto* covMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) covMatrix, Form("Covariance_Eigen_%s_TH2D", parSet.getName().c_str()));
      auto* corMatrix = GenericToolbox::convertToCorrelationMatrix((TMatrixD*) covMatrix);
      auto* corMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D(corMatrix, Form("Correlation_Eigen_%s_TH2D", parSet.getName().c_str()));

      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        covMatrixTH2D->GetXaxis()->SetBinLabel(1+iEigen, (parSet.getName() + "/eigen_#" + std::to_string(iEigen)).c_str());
        covMatrixTH2D->GetYaxis()->SetBinLabel(1+iEigen, (parSet.getName() + "/eigen_#" + std::to_string(iEigen)).c_str());
        corMatrixTH2D->GetXaxis()->SetBinLabel(1+iEigen, (parSet.getName() + "/eigen_#" + std::to_string(iEigen)).c_str());
        corMatrixTH2D->GetYaxis()->SetBinLabel(1+iEigen, (parSet.getName() + "/eigen_#" + std::to_string(iEigen)).c_str());

        totalCovTH2D->GetXaxis()->SetBinLabel(1 + parameterIndexOffset + iEigen, (parSet.getName() + "/eigen_#" + std::to_string(iEigen)).c_str());
        totalCovTH2D->GetYaxis()->SetBinLabel(1 + parameterIndexOffset + iEigen, (parSet.getName() + "/eigen_#" + std::to_string(iEigen)).c_str());
      }

      GenericToolbox::mkdirTFile(parSetDir, "matrices")->cd();
      covMatrix->Write("Covariance_Eigen_TMatrixD");
      covMatrixTH2D->Write("Covariance_Eigen_TH2D");
      corMatrix->Write("Correlation_Eigen_TMatrixD");
      corMatrixTH2D->Write("Correlation_Eigen_TH2D");

      auto* originalCovMatrix = new TMatrixD(int(parSet.getParameterList().size()), int(parSet.getParameterList().size()));
      (*originalCovMatrix) =  (*parSet.getEigenVectors());
      (*originalCovMatrix) *= (*covMatrix);
      (*originalCovMatrix) *= (*parSet.getInvertedEigenVectors());
      covMatrix = originalCovMatrix;

    }

    auto* covMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) covMatrix, Form("Covariance_%s_TH2D", parSet.getName().c_str()));
    auto* corMatrix = GenericToolbox::convertToCorrelationMatrix((TMatrixD*) covMatrix);
    auto* corMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D(corMatrix, Form("Correlation_%s_TH2D", parSet.getName().c_str()));

    for( const auto& par : parSet.getParameterList() ){
      covMatrixTH2D->GetXaxis()->SetBinLabel(1+par.getParameterIndex(), (parSet.getName() + "/" + par.getTitle()).c_str());
      covMatrixTH2D->GetYaxis()->SetBinLabel(1+par.getParameterIndex(), (parSet.getName() + "/" + par.getTitle()).c_str());
      corMatrixTH2D->GetXaxis()->SetBinLabel(1+par.getParameterIndex(), (parSet.getName() + "/" + par.getTitle()).c_str());
      corMatrixTH2D->GetYaxis()->SetBinLabel(1+par.getParameterIndex(), (parSet.getName() + "/" + par.getTitle()).c_str());
    }

    GenericToolbox::mkdirTFile(parSetDir, "matrices")->cd();
    covMatrix->Write("Covariance_TMatrixD");
    covMatrixTH2D->Write("Covariance_TH2D");
    corMatrix->Write("Correlation_TMatrixD");
    corMatrixTH2D->Write("Correlation_TH2D");

    // Parameters
    GenericToolbox::mkdirTFile(parSetDir, "values")->cd();
    auto* postFitErrorHist = new TH1D("postFitErrors_TH1D", "Post-fit Errors", parSet.getNbParameters(), 0, parSet.getNbParameters());
    auto* preFitErrorHist = new TH1D("preFitErrors_TH1D", "Pre-fit Errors", parSet.getNbParameters(), 0, parSet.getNbParameters());
    for( const auto& par : parSet.getParameterList() ){
      postFitErrorHist->GetXaxis()->SetBinLabel(1 + par.getParameterIndex(), par.getTitle().c_str());
      postFitErrorHist->SetBinContent( 1 + par.getParameterIndex(), par.getParameterValue());
      postFitErrorHist->SetBinError( 1 + par.getParameterIndex(), TMath::Sqrt((*covMatrix)[par.getParameterIndex()][par.getParameterIndex()]));

      preFitErrorHist->GetXaxis()->SetBinLabel(1 + par.getParameterIndex(), par.getTitle().c_str());
      preFitErrorHist->SetBinContent( 1 + par.getParameterIndex(), par.getPriorValue() );
      if( par.isEnabled() and not par.isFixed() ){

        double priorFraction = TMath::Sqrt((*covMatrix)[par.getParameterIndex()][par.getParameterIndex()]) / par.getStdDevValue();

        std::stringstream ss;

        if( priorFraction < 1E-2 ) ss << GenericToolbox::ColorCodes::yellowBackGround;
        if( priorFraction > 1 ) ss << GenericToolbox::ColorCodes::redBackGround;

        ss << "Postfit error of \"" << parSet.getName() << "/" << par.getTitle() << "\": "
           << TMath::Sqrt((*covMatrix)[par.getParameterIndex()][par.getParameterIndex()])
           << " (" << priorFraction * 100
           << "% of the prior)" << GenericToolbox::ColorCodes::resetColor
           << std::endl;

        LogInfo << ss.str();

        preFitErrorHist->SetBinError( 1 + par.getParameterIndex(), par.getStdDevValue() );
      }
    }

    if( not gStyle->GetCanvasPreferGL() ){
      preFitErrorHist->SetFillColor(kRed-9);
    }
    else{
      preFitErrorHist->SetFillColorAlpha(kRed-9, 0.7);
    }

//        preFitErrorHist->SetFillColorAlpha(kRed-9, 0.5);
//        preFitErrorHist->SetFillStyle(4050); // 50 % opaque ?
    preFitErrorHist->SetMarkerStyle(kFullDotLarge);
    preFitErrorHist->SetMarkerColor(kRed-3);
    preFitErrorHist->SetTitle(Form("Pre-fit Errors of %s", parSet.getName().c_str()));
    preFitErrorHist->Write();

    postFitErrorHist->SetLineColor(9);
    postFitErrorHist->SetLineWidth(2);
    postFitErrorHist->SetMarkerColor(9);
    postFitErrorHist->SetMarkerStyle(kFullDotLarge);
    postFitErrorHist->SetTitle(Form("Post-fit Errors of %s", parSet.getName().c_str()));
    postFitErrorHist->Write();

    auto* errorsCanvas = new TCanvas(Form("Fit Constraints for %s", parSet.getName().c_str()), Form("Fit Constraints for %s", parSet.getName().c_str()), 800, 600);
    errorsCanvas->cd();

    preFitErrorHist->SetMarkerSize(0);
    preFitErrorHist->Draw("E2");

    TH1D preFitErrorHistLine = TH1D("preFitErrorHistLine", "preFitErrorHistLine",
                                    preFitErrorHist->GetNbinsX(),
                                    preFitErrorHist->GetXaxis()->GetXmin(),
                                    preFitErrorHist->GetXaxis()->GetXmax()
                                    );
    GenericToolbox::transformBinContent(&preFitErrorHistLine, [&](TH1D* h_, int b_){
      h_->SetBinContent(b_, preFitErrorHist->GetBinContent(b_));
    });


    preFitErrorHistLine.SetLineColor(kRed-3);
    preFitErrorHistLine.Draw("SAME");

    errorsCanvas->Update(); // otherwise does not display...
    postFitErrorHist->Draw("E1 X0 SAME");

    gPad->SetGridx();
    gPad->SetGridy();

    preFitErrorHist->SetTitle(Form("Pre-fit/Post-fit Comparison for %s", parSet.getName().c_str()));
    errorsCanvas->Write("fitConstraints_TCanvas");




    if( not parSet.isUseEigenDecompInFit() ){
      parameterIndexOffset += int(parSet.getNbParameters());
    }
    else{
      parameterIndexOffset += parSet.getNbEnabledEigenParameters();
    }

  } // parSet

}

double FitterEngine::fetchCurrentParameterValue(int iFitPar_) {
  int iFitPar = 0;
  for( const auto& parSet : _propagator_.getParameterSetsList() ){
    if( not parSet.isUseEigenDecompInFit() ){
      for( const auto& par : parSet.getParameterList() ){
        if( iFitPar++ == iFitPar_ ) return par.getParameterValue();
      }
    }
    else{
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        if( iFitPar++ == iFitPar_ ) return parSet.getEigenParameterValue(iEigen);
      }
    }
  }
  return std::nan("parameter not found");
}
void FitterEngine::updateParameterValue(int iFitPar_, double parameterValue_) {
  int iFitPar = 0;
  for( auto& parSet : _propagator_.getParameterSetsList() ){
    if( not parSet.isUseEigenDecompInFit() ){
      for( auto& par : parSet.getParameterList() ){
        if( iFitPar++ == iFitPar_ ) par.setParameterValue(parameterValue_);
      }
    }
    else{
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        if( iFitPar++ == iFitPar_ ) parSet.setEigenParameter(iEigen, parameterValue_);
      }
    }
  }
}

void FitterEngine::rescaleParametersStepSize(){
  LogDebug << __METHOD_NAME__ << std::endl;

  updateChi2Cache();
  double baseChi2Pull = _chi2PullsBuffer_;
  double baseChi2 = _chi2Buffer_;

  // +1 sigma
  int iFitPar = -1;
  for( auto& parSet : _propagator_.getParameterSetsList() ){


    if( parSet.isUseEigenDecompInFit() ){
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        iFitPar++;

        double currentParValue = parSet.getEigenParameterValue(iEigen);
        parSet.setEigenParameter(iEigen, currentParValue + parSet.getEigenSigma(iEigen));
        parSet.propagateEigenToOriginal();

        updateChi2Cache();
        double deltaChi2 = _chi2Buffer_ - baseChi2;
        double deltaChi2Pulls = _chi2PullsBuffer_ - baseChi2Pull;

//        double stepSize = TMath::Sqrt(deltaChi2Pulls)/TMath::Sqrt(deltaChi2);
        double stepSize = 1./TMath::Sqrt(std::abs(deltaChi2));

        LogInfo << "Step size of " << parSet.getName() + "/eigen_#" << iEigen
                << " -> σ x " << _parStepGain_ << " x " << stepSize
                << " -> Δχ² = " << deltaChi2 << " = " << deltaChi2 - deltaChi2Pulls << "(stat) + " << deltaChi2Pulls << "(pulls)" << std::endl;

        stepSize *= parSet.getEigenSigma(iEigen) * _parStepGain_;

        parSet.setEigenParStepSize(iEigen, stepSize);
        parSet.setEigenParameter(iEigen, currentParValue);
        parSet.propagateEigenToOriginal();
      }
    }
    else{
      for( auto& par : parSet.getParameterList() ){
        iFitPar++;

        if( not par.isEnabled() ){
          continue;
        }

        double currentParValue = par.getParameterValue();
        par.setParameterValue( currentParValue + par.getStdDevValue() );

        updateChi2Cache();

        double deltaChi2 = _chi2Buffer_ - baseChi2;
        double deltaChi2Pulls = _chi2PullsBuffer_ - baseChi2Pull;

        // Consider a parabolic approx:
        // only rescale with X2 stat?
//        double stepSize = TMath::Sqrt(deltaChi2Pulls)/TMath::Sqrt(deltaChi2);

        // full rescale
        double stepSize = 1./TMath::Sqrt(std::abs(deltaChi2));

        LogInfo << "Step size of " << parSet.getName() + "/" + par.getTitle()
            << " -> σ x " << _parStepGain_ << " x " << stepSize
            << " -> Δχ² = " << deltaChi2 << " = " << deltaChi2 - deltaChi2Pulls << "(stat) + " << deltaChi2Pulls << "(pulls)";

        stepSize *= par.getStdDevValue() * _parStepGain_;

        par.setStepSize( stepSize );
        par.setParameterValue( currentParValue + stepSize );
        updateChi2Cache();
        LogInfo << " -> Δχ²(step) = " << _chi2Buffer_ - baseChi2 << std::endl;
        par.setParameterValue( currentParValue );
      }
    }

  }

  LogInfo << "Reupdating chi2..." << std::endl;
  updateChi2Cache();
  LogDebug << "END" << std::endl;

}
void FitterEngine::initializeMinimizer(bool doReleaseFixed_){
  LogInfo << __METHOD_NAME__ << std::endl;

  _minimizerConfig_ = JsonUtils::fetchValue(_config_, "minimizerConfig", nlohmann::json());
  JsonUtils::forwardConfig(_minimizerConfig_);

  _minimizerType_ = JsonUtils::fetchValue(_minimizerConfig_, "minimizer", "Minuit2");
  _minimizerAlgo_ = JsonUtils::fetchValue(_minimizerConfig_, "algorithm", "");

  _useNormalizedFitSpace_ = JsonUtils::fetchValue(_minimizerConfig_, "useNormalizedFitSpace", true);

  _minimizer_ = std::shared_ptr<ROOT::Math::Minimizer>(
      ROOT::Math::Factory::CreateMinimizer(_minimizerType_, _minimizerAlgo_)
  );

  LogThrowIf(_minimizer_ == nullptr, "Could not create minimizer: " << _minimizerType_ << "/" << _minimizerAlgo_)

  if( _minimizerAlgo_.empty() ) _minimizerAlgo_ = _minimizer_->Options().MinimizerAlgorithm();

  _functor_ = std::shared_ptr<ROOT::Math::Functor>(
      new ROOT::Math::Functor(
          this, &FitterEngine::evalFit, _nbFitParameters_
      )
  );

  _minimizer_->SetFunction(*_functor_);
  _minimizer_->SetStrategy(JsonUtils::fetchValue(_minimizerConfig_, "strategy", 1));
  _minimizer_->SetPrintLevel(JsonUtils::fetchValue(_minimizerConfig_, "print_level", 2));
  _minimizer_->SetTolerance(JsonUtils::fetchValue(_minimizerConfig_, "tolerance", 1E-4));
  _minimizer_->SetMaxIterations(JsonUtils::fetchValue(_minimizerConfig_, "max_iter", (unsigned int)(500) ));
  _minimizer_->SetMaxFunctionCalls(JsonUtils::fetchValue(_minimizerConfig_, "max_fcn", (unsigned int)(1E9)));

  int iPar = -1;
  for( auto& parSet : _propagator_.getParameterSetsList() ){

    if( not parSet.isUseEigenDecompInFit() ){
      for( auto& par : parSet.getParameterList()  ){
        iPar++;
        if(not _useNormalizedFitSpace_){
          _minimizer_->SetVariable( iPar,parSet.getName() + "/" + par.getTitle(), par.getParameterValue(),par.getStepSize() );
          if(par.getMinValue() == par.getMinValue()){ _minimizer_->SetVariableLowerLimit(iPar, par.getMinValue()); }
          if(par.getMaxValue() == par.getMaxValue()){ _minimizer_->SetVariableUpperLimit(iPar, par.getMaxValue()); }
          // Changing the boundaries, change the value/step size?
          _minimizer_->SetVariableValue(iPar, par.getParameterValue());
          _minimizer_->SetVariableStepSize(iPar, par.getStepSize());
        }
        else{
          _minimizer_->SetVariable( iPar,parSet.getName() + "/" + par.getTitle(),
                                    FitParameterSet::toNormalizedParValue(par.getParameterValue(), par),
                                    FitParameterSet::toNormalizedParRange(par.getStepSize(), par)
                                    );
          if(par.getMinValue() == par.getMinValue()){ _minimizer_->SetVariableLowerLimit(iPar, FitParameterSet::toNormalizedParValue(par.getMinValue(), par)); }
          if(par.getMaxValue() == par.getMaxValue()){ _minimizer_->SetVariableUpperLimit(iPar, FitParameterSet::toNormalizedParValue(par.getMaxValue(), par)); }
          // Changing the boundaries, change the value/step size?
          _minimizer_->SetVariableValue(iPar, FitParameterSet::toNormalizedParValue(par.getParameterValue(), par));
          _minimizer_->SetVariableStepSize(iPar, FitParameterSet::toNormalizedParRange(par.getStepSize(), par));
        }


        if( not doReleaseFixed_ or not JsonUtils::fetchValue(parSet.getConfig(), "releaseFixedParametersOnHesse", true) ){
          if( not par.isEnabled() or par.isFixed() ) _minimizer_->FixVariable(iPar);
        }
      } // par
    }
    else{
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        iPar++;
        if(not _useNormalizedFitSpace_){
          _minimizer_->SetVariable( iPar,parSet.getName() + "/eigen_#" + std::to_string(iEigen),
                                    parSet.getEigenParameterValue(iEigen),
                                    parSet.getEigenParStepSize(iEigen)
          );
        }
        else{
          _minimizer_->SetVariable( iPar,parSet.getName() + "/eigen_#" + std::to_string(iEigen),
                                    parSet.toNormalizedEigenParValue(parSet.getEigenParameterValue(iEigen),iEigen),
                                    parSet.toNormalizedEigenParRange(parSet.getEigenParStepSize(iEigen), iEigen)
          );
        }

        if( not doReleaseFixed_ or not JsonUtils::fetchValue(parSet.getConfig(), "releaseFixedParametersOnHesse", true) ){
          if( parSet.isEigenParFixed(iEigen) ) {
            _minimizer_->FixVariable(iPar);
          }
        }
      }
    }

  } // parSet

}
