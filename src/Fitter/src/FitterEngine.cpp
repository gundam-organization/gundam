//
// Created by Nadrino on 11/06/2021.
//

#include <Math/Factory.h>
#include "TGraph.h"

#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.RawDataArray.h"

#include "JsonUtils.h"
#include "FitterEngine.h"
#include "GlobalVariables.h"

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
  while( _config_.is_string() ){
    LogWarning << "Forwarding " << __CLASS_NAME__ << " config: \"" << _config_.get<std::string>() << "\"" << std::endl;
    _config_ = JsonUtils::readConfigFile(_config_.get<std::string>());
  }
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

  if( JsonUtils::fetchValue(_config_, "enableParStepRescale", true) ){
    _parStepScale_ = JsonUtils::fetchValue(_config_, "parStepScale", _parStepScale_);
    LogInfo << "Using parameter step scale: " << _parStepScale_ << std::endl;
    this->rescaleParametersStepSize();
  }


  if( JsonUtils::fetchValue(_config_, "fixGhostFitParameters", false) ) this->fixGhostFitParameters();

  _convergenceMonitor_.addDisplayedQuantity("VarName");
  _convergenceMonitor_.addDisplayedQuantity("LastAddedValue");
  _convergenceMonitor_.addDisplayedQuantity("SlopePerCall");

  _convergenceMonitor_.getQuantity("VarName").title = "Chi2";
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

  this->initializeMinimizer();

  if( JsonUtils::fetchValue(_config_, "throwMcBeforeFit", false) ){
    LogInfo << "Throwing MC parameters (uncorrelated) away from their prior..." << std::endl;
    double throwGain = JsonUtils::fetchValue(_config_, "throwMcBeforeFitGain", 1.);
    LogInfo << "Throw gain set to: " << throwGain << std::endl;
    this->throwMcParameters(throwGain);
  }

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
const Propagator &FitterEngine::getPropagator() const {
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
  double baseChi2Stat = _chi2StatBuffer_;

  // +1 sigma
  int iFitPar = -1;
  std::stringstream ssPrint;
  for( auto& parSet : _propagator_.getParameterSetsList() ){

    if( not JsonUtils::fetchValue(parSet.getConfig(), "fixGhostFitParameters", false) ) {
      LogWarning << "Skipping \"" << parSet.getName() << "\" as fixGhostFitParameters is set to false" << std::endl;
      if( not parSet.isUseEigenDecompInFit() ) iFitPar += parSet.getNbParameters();
      else iFitPar += parSet.getNbEnabledEigenParameters();
      continue;
    }

    if( parSet.isUseEigenDecompInFit() ){
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        iFitPar++;
        ssPrint.str("");

        double currentParValue = parSet.getEigenParameterValue(iEigen);
        parSet.setEigenParameter(iEigen, currentParValue + parSet.getEigenSigma(iEigen));
        parSet.propagateEigenToOriginal();

        ssPrint << "(" << iFitPar+1 << "/" << _nbFitParameters_ << ") +1σ on " << parSet.getName() + "/eigen_#" + std::to_string(iEigen)
                << " -> " << parSet.getEigenParameterValue(iEigen);
        LogInfo << ssPrint.str() << "..." << std::endl;

        updateChi2Cache();

        double deltaChi2 = _chi2StatBuffer_ - baseChi2Stat;
        ssPrint << ": Δχ²(stat) = " << std::fabs(deltaChi2);

        LogInfo.moveTerminalCursorBack(1);
        LogInfo << ssPrint.str() << std::endl;

        if( std::fabs(deltaChi2) < JsonUtils::fetchValue(_config_, "ghostParameterDeltaChi2Threshold", 1E-6) ){
          parSet.setEigenParIsFixed(iEigen, true);

          ssPrint << " < " << JsonUtils::fetchValue(_config_, "ghostParameterDeltaChi2Threshold", 1E-6) << " -> " << "FIXED";
          LogInfo.moveTerminalCursorBack(1);
          LogInfo << GenericToolbox::ColorCodes::redBackGround << ssPrint.str() << GenericToolbox::ColorCodes::resetColor << std::endl;
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
                  << " -> " << par.getParameterValue();
          LogInfo << ssPrint.str() << "..." << std::endl;

          updateChi2Cache();
          double deltaChi2 = _chi2StatBuffer_ - baseChi2Stat;
          ssPrint << ": Δχ²(stat) = " << std::fabs(deltaChi2);

          LogInfo.moveTerminalCursorBack(1);
          LogInfo << ssPrint.str() << std::endl;

          if( std::fabs(deltaChi2) < JsonUtils::fetchValue(_config_, "ghostParameterDeltaChi2Threshold", 1E-6) ){
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

void FitterEngine::throwMcParameters(double gain_) {
  LogInfo << __METHOD_NAME__ << std::endl;

  // TODO: IMPLEMENT CHOLESKY DECOMP
  int iFitPar = -1;
  for( auto& parSet : _propagator_.getParameterSetsList() ){

    if(not parSet.isEnabled()) continue;

    if( not parSet.isEnableThrowMcBeforeFit() ){
      LogWarning << "\"" << parSet.getName() << "\" has marked disabled throwMcBeforeFit: skipping." << std::endl;
      if( not parSet.isUseEigenDecompInFit() ) iFitPar += int(parSet.getParameterList().size());
      else iFitPar += int(parSet.getNbEnabledEigenParameters());

      for(auto& entry : JsonUtils::fetchValue(parSet.getConfig(), "moveMcParBeforeFit", std::vector<nlohmann::json>())){
        int parIndex = JsonUtils::fetchValue<int>(entry, "parIndex");

        double pushVal;
        if( parSet.isUseEigenDecompInFit() ){
          pushVal =
              parSet.getEigenParameterValue(parIndex)
              + parSet.getEigenSigma(parIndex)
                * JsonUtils::fetchValue<double>(entry, "nbSigmaAway");


          LogWarning << "Pushing eigen_#" << parIndex << " to " << pushVal << std::endl;
          parSet.setEigenParameter(parIndex, pushVal);
        }
        else{
          pushVal =
              parSet.getParameterList()[parIndex].getParameterValue()
              + parSet.getParameterList()[parIndex].getStdDevValue()
                * JsonUtils::fetchValue<double>(entry, "nbSigmaAway");


          LogWarning << "Pushing #" << parIndex << " to " << pushVal << std::endl;
          parSet.getParameterList()[parIndex].setParameterValue( pushVal );
        }

      }

      continue;
    }
    else{
      LogWarning << "Throwing correlated parameters for \"" << parSet.getName() << "\"" << std::endl;
    }


    if( not parSet.isUseEigenDecompInFit() ){

      auto throws = GenericToolbox::throwCorrelatedParameters(
          GenericToolbox::getCholeskyMatrix(parSet.getOriginalCovarianceMatrix())
      );

      int iPar{-1};
      for( auto& par : parSet.getParameterList() ){
        iFitPar++; iPar++;

        if( not _minimizer_->IsFixedVariable(iFitPar) ){
          LogInfo << "Throwing par " << par.getTitle() << ": " << par.getParameterValue();
          par.setParameterValue(
              par.getPriorValue()
              + gain_ * throws[iPar] );
          LogInfo << " → " << par.getParameterValue() << std::endl;
          _minimizer_->SetVariableValue(iFitPar, par.getParameterValue() );
        }
      }
    }
    else{
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        iFitPar++;
        if( not parSet.isEnableThrowMcBeforeFit() ) continue;
        if( not _minimizer_->IsFixedVariable(iFitPar) ){
          LogInfo << "Throwing eigen #" << iEigen << ": " << parSet.getEigenParameterValue(iEigen);
          parSet.setEigenParameter(
              iEigen,
              parSet.getEigenParameterValue(iEigen)
              + gain_ * _prng_.Gaus(0, parSet.getEigenSigma(iEigen) )
          );
          LogInfo << " → " << parSet.getEigenParameterValue(iEigen) << std::endl;
          _minimizer_->SetVariableValue(iFitPar, parSet.getEigenParameterValue(iEigen) );

          // placeholder
        }
      }
    }

  }

  _propagator_.preventRfPropagation(); // Making sure since we need the weight of each event
  _propagator_.propagateParametersOnSamples();
}

void FitterEngine::fit(){
  LogWarning << __METHOD_NAME__ << std::endl;

  LogWarning << "─────────────────────────────" << std::endl;
  LogWarning << "Summary of the fit parameters" << std::endl;
  LogWarning << "─────────────────────────────" << std::endl;
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

  LogInfo << "Number of defined parameters: " << _minimizer_->NDim() << std::endl
          << "Number of free parameters   : " << _minimizer_->NFree() << std::endl
          << "Number of fixed parameters  : " << _minimizer_->NDim() - _minimizer_->NFree()
          << std::endl;

  _propagator_.allowRfPropagation(); // if RF are setup -> a lot faster
  updateChi2Cache();

  LogWarning << "───────────────────" << std::endl;
  LogWarning << "Calling minimize..." << std::endl;
  LogWarning << "───────────────────" << std::endl;
  _fitUnderGoing_ = true;
  _fitHasConverged_ = _minimizer_->Minimize();
  int nbMinimizeCalls = _nbFitCalls_;
  LogInfo << _convergenceMonitor_.generateMonitorString(); // lasting printout
  LogInfo << "Minimization ended after " << nbMinimizeCalls << " calls." << std::endl;
  LogWarning << "Status code: " << minuitStatusCodeStr.at(_minimizer_->Status()) << std::endl;
  LogWarning << "Covariance matrix status code: " << covMatrixStatusCodeStr.at(_minimizer_->CovMatrixStatus()) << std::endl;
  if( _saveDir_ != nullptr ){
    GenericToolbox::mkdirTFile(_saveDir_, "fit")->cd();
    _chi2HistoryTree_->Write();
  }

  if( _fitHasConverged_ ){
    LogInfo << "Fit converged!" << std::endl;

    if( _enablePostFitScan_ ){
      _fitUnderGoing_ = false;
      LogInfo << "Scanning parameters around the minimum point..." << std::endl;
      this->scanParameters(-1, "postFit/scan");
      _fitUnderGoing_ = true;
    }

    LogInfo << "Evaluating post-fit errors..." << std::endl;

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

        LogWarning << "────────────────" << std::endl;
        LogWarning << "Calling MINOS..." << std::endl;
        LogWarning << "────────────────" << std::endl;

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
//      LogInfo << "Releasing constraints for HESSE..." << std::endl;
//      initializeMinimizer(true);

        LogWarning << "────────────────" << std::endl;
        LogWarning << "Calling HESSE..." << std::endl;
        LogWarning << "────────────────" << std::endl;
        LogInfo << "Number of defined parameters: " << _minimizer_->NDim() << std::endl
                << "Number of free parameters   : " << _minimizer_->NFree() << std::endl
                << "Number of fixed parameters  : " << _minimizer_->NDim() - _minimizer_->NFree()
                << std::endl;
        _fitHasConverged_ = _minimizer_->Hesse();
        LogInfo << "Hesse ended after " << _nbFitCalls_ - nbMinimizeCalls << " calls." << std::endl;
        LogWarning << "HESSE status code: " << hesseStatusCodeStr.at(_minimizer_->Status()) << std::endl;
        LogWarning << "Covariance matrix status code: " << covMatrixStatusCodeStr.at(_minimizer_->CovMatrixStatus()) << std::endl;
//        LogWarning << "Did minuit performed a detailed error validation? " << GET_VAR_NAME_VALUE(_minimizer_->IsValidError()) << std::endl;

        if(not _fitHasConverged_){
          LogError  << "Hesse did not converge." << std::endl;
        }
        else{
          LogInfo << "Hesse converged." << std::endl;
        }

      }
      else{
        LogError << GET_VAR_NAME_VALUE(errorAlgo) << " not implemented." << std::endl;
      }
    }

    LogWarning << _convergenceMonitor_.generateMonitorString(); // lasting printout
  }
  else{
    LogError << "Did not converged." << std::endl;
    LogError << _convergenceMonitor_.generateMonitorString(); // lasting printout
  }

  _propagator_.preventRfPropagation(); // since we need the weight of each event
  _propagator_.propagateParametersOnSamples();

  _fitUnderGoing_ = false;
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
  _nbFitCalls_++;

  // Update fit parameter values:
  int iFitPar = -1;
  for( auto& parSet : _propagator_.getParameterSetsList() ){
    if( not parSet.isUseEigenDecompInFit() ){
      for( auto& par : parSet.getParameterList() ){
        iFitPar++;
        par.setParameterValue( parArray_[iFitPar] );
      }
    }
    else{
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        iFitPar++;
        parSet.setEigenParameter(iEigen, parArray_[iFitPar]);
      }
      parSet.propagateEigenToOriginal();
    }
  }

  // Compute the Chi2
  updateChi2Cache();

  _evalFitAvgTimer_.counts++; _evalFitAvgTimer_.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);

  if( _convergenceMonitor_.isGenerateMonitorStringOk() and _fitUnderGoing_ ){
    std::stringstream ss;
    ss << __METHOD_NAME__ << ": call #" << _nbFitCalls_ << std::endl;
    ss << "Avg χ² computation time: " << _evalFitAvgTimer_ << std::endl;
    if( not _propagator_.isUseResponseFunctions() ){
      ss << "├─ Current RAM: " << GenericToolbox::parseSizeUnits(GenericToolbox::getProcessMemoryUsage()) << std::endl;
//      if(GlobalVariables::isEnableDevMode()) ss << "├─ Avg time to update dial caches: " << _propagator_.dialUpdate << std::endl;
      ss << "├─ Avg time to propagate weights: " << _propagator_.weightProp << std::endl;
      ss << "├─ Avg time to fill histograms: " << _propagator_.fillProp;
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
  }

  // Fill History
  _chi2HistoryTree_->Fill();
//  _chi2History_["Total"].emplace_back(_chi2Buffer_);
//  _chi2History_["Stat"].emplace_back(_chi2StatBuffer_);
//  _chi2History_["Syst"].emplace_back(_chi2PullsBuffer_);

  return _chi2Buffer_;
}

void FitterEngine::writePostFitData() {
  LogInfo << __METHOD_NAME__ << std::endl;

  LogThrowIf(not _fitIsDone_, "Can't do " << __METHOD_NAME__ << " while fit has not been called.")

  if( _saveDir_ == nullptr ){
    LogError << "_saveDir_ not set, won't save post fit data." << std::endl;
    return;
  }

  auto* postFitDir = GenericToolbox::mkdirTFile(_saveDir_, "postFit");

  this->generateSamplePlots("postFit/samples");

  auto* errorDir = GenericToolbox::mkdirTFile(postFitDir, "errors");

  double covarianceMatrixArray[_minimizer_->NDim() * _minimizer_->NDim()];
  _minimizer_->GetCovMatrix(covarianceMatrixArray);
  TMatrixDSym fitterCovarianceMatrix(int(_minimizer_->NDim()), covarianceMatrixArray);
  TH2D* fitterCovarianceMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) &fitterCovarianceMatrix, "fitterCovarianceMatrixTH2D");

  LogInfo << "Fitter covariance matrix is " << fitterCovarianceMatrix.GetNrows() << "x" << fitterCovarianceMatrix.GetNcols() << std::endl;


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
              fitterCovarianceMatrix[parameterIndexOffset + parRow.getParameterIndex()][parameterIndexOffset +
                                                                                        parCol.getParameterIndex()];
        } // par Y
      } // par X

      for( const auto& par : parSet.getParameterList() ) {
        fitterCovarianceMatrixTH2D->GetXaxis()->SetBinLabel(1 + parameterIndexOffset + par.getParameterIndex(),
                                                            (parSet.getName() + "/" + par.getTitle()).c_str());
        fitterCovarianceMatrixTH2D->GetYaxis()->SetBinLabel(1 + parameterIndexOffset + par.getParameterIndex(),
                                                            (parSet.getName() + "/" + par.getTitle()).c_str());
      }
    }
    else{

      covMatrix = new TMatrixD(parSet.getNbEnabledEigenParameters(), parSet.getNbEnabledEigenParameters());
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        for( int jEigen = 0 ; jEigen < parSet.getNbEnabledEigenParameters() ; jEigen++ ){
          if( parSet.isEigenParFixed(iEigen) or parSet.isEigenParFixed(jEigen) ){
            if( iEigen == jEigen ){
              (*covMatrix)[iEigen][jEigen] = parSet.getEigenSigma(iEigen)*parSet.getEigenSigma(iEigen);
            }
            else{
              (*covMatrix)[iEigen][jEigen] = 0;
            }
          }
          else{
            (*covMatrix)[iEigen][jEigen] = fitterCovarianceMatrix[parameterIndexOffset + iEigen][parameterIndexOffset + jEigen];
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

        fitterCovarianceMatrixTH2D->GetXaxis()->SetBinLabel(1+parameterIndexOffset+iEigen, (parSet.getName() + "/eigen_#" + std::to_string(iEigen)).c_str());
        fitterCovarianceMatrixTH2D->GetYaxis()->SetBinLabel(1+parameterIndexOffset+iEigen, (parSet.getName() + "/eigen_#" + std::to_string(iEigen)).c_str());
      }

      GenericToolbox::mkdirTFile(parSetDir, "matrices")->cd();
      covMatrix->Write("Covariance_Eigen_TMatrixD");
      covMatrixTH2D->Write("Covariance_Eigen_TH2D");
      corMatrix->Write("Correlation_Eigen_TMatrixD");
      corMatrixTH2D->Write("Correlation_Eigen_TH2D");

      auto* originalCovMatrix = new TMatrixD(int(parSet.getParameterList().size()), int(parSet.getParameterList().size()));
//      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
//        for( int jEigen = 0 ; jEigen < parSet.getNbEnabledEigenParameters() ; jEigen++ ){
//          (*originalCovMatrix)[iEigen][jEigen] = (*covMatrix)[iEigen][jEigen];
//        }
//      }

      (*originalCovMatrix) = (*parSet.getInvertedEigenVectors());
      (*originalCovMatrix) *= (*covMatrix);
      (*originalCovMatrix) *= (*parSet.getEigenVectors());

//      (*originalCovMatrix) = (*parSet.getInvertedEigenVectors()) * (*originalCovMatrix) * (*parSet.getEigenVectors());

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

    auto* errorsCanvas = new TCanvas(Form("Fit Constraints for %s", parSet.getName().c_str()), Form("Fit Constraints for %s", parSet.getName().c_str()), 800, 600);
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


    if( not parSet.isUseEigenDecompInFit() ){
      parameterIndexOffset += int(parSet.getNbParameters());
    }
    else{
      parameterIndexOffset += parSet.getNbEnabledEigenParameters();
    }

  } // parSet

  errorDir->cd();
  fitterCovarianceMatrix.Write("fitterCovarianceMatrix_TMatrixDSym");
  fitterCovarianceMatrixTH2D->Write("fitterCovarianceMatrix_TH2D");

  LogInfo << "Fetching Hessian matrix..." << std::endl;
  double hessianMatrixArray[_minimizer_->NDim() * _minimizer_->NDim()];
  _minimizer_->GetHessianMatrix(hessianMatrixArray);
  TMatrixDSym fitterHessianMatrix(int(_minimizer_->NDim()), hessianMatrixArray);

  LogInfo << "Decomposing Hessian matrix..." << std::endl;
  auto decompHessian = TMatrixDSymEigen(fitterHessianMatrix);

  fitterHessianMatrix.Write("fitterHessianMatrix_TMatrixDSym");
  auto* fitterHessianMatrix_TH2D = GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*)&fitterHessianMatrix);
  fitterHessianMatrix_TH2D->Write("fitterHessianMatrix_TH2D");

  decompHessian.GetEigenValues().Write("fitterHessianEigen_TVectorD");
  auto* fitterHessianEigen_TH1D = GenericToolbox::convertTVectorDtoTH1D(&decompHessian.GetEigenValues());
  fitterHessianEigen_TH1D->Write("fitterHessianEigen_TH1D");

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
        double stepSize = 1./TMath::Sqrt(std::fabs(deltaChi2));

        LogInfo << "Step size of " << parSet.getName() + "/eigen_#" << iEigen
                << " -> σ x " << _parStepScale_ << " x " << stepSize
                << " -> Δχ² = " << deltaChi2 << " = " << deltaChi2 - deltaChi2Pulls << "(stat) + " << deltaChi2Pulls << "(pulls)" << std::endl;

        stepSize *= parSet.getEigenSigma(iEigen) * _parStepScale_;

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
        double stepSize = 1./TMath::Sqrt(std::fabs(deltaChi2));

        LogInfo << "Step size of " << parSet.getName() + "/" + par.getTitle()
            << " -> σ x " << _parStepScale_ << " x " << stepSize
            << " -> Δχ² = " << deltaChi2 << " = " << deltaChi2 - deltaChi2Pulls << "(stat) + " << deltaChi2Pulls << "(pulls)";

        stepSize *= par.getStdDevValue() * _parStepScale_;

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
  if( _minimizerConfig_.is_string() ){ _minimizerConfig_ = JsonUtils::readConfigFile(_minimizerConfig_.get<std::string>()); }

  _minimizer_ = std::shared_ptr<ROOT::Math::Minimizer>(
      ROOT::Math::Factory::CreateMinimizer(
          JsonUtils::fetchValue(_minimizerConfig_, "minimizer", "Minuit2"),
          JsonUtils::fetchValue(_minimizerConfig_, "algorithm", "Migrad")
      )
  );

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
        _minimizer_->SetVariable( iPar,parSet.getName() + "/" + par.getTitle(), par.getParameterValue(),par.getStepSize() );
        if(par.getMinValue() == par.getMinValue()){ _minimizer_->SetVariableLowerLimit(iPar, par.getMinValue()); }
        if(par.getMaxValue() == par.getMaxValue()){ _minimizer_->SetVariableUpperLimit(iPar, par.getMaxValue()); }
        _minimizer_->SetVariableValue(iPar, par.getParameterValue());
        _minimizer_->SetVariableStepSize(iPar, par.getStepSize());

        if( not doReleaseFixed_ or not JsonUtils::fetchValue(parSet.getConfig(), "releaseFixedParametersOnHesse", true) ){
          if( not par.isEnabled() or par.isFixed() ) _minimizer_->FixVariable(iPar);
        }
      } // par
    }
    else{
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        iPar++;
        _minimizer_->SetVariable( iPar,parSet.getName() + "/eigen_#" + std::to_string(iEigen),
                                  parSet.getEigenParameterValue(iEigen),
                                  parSet.getEigenParStepSize(iEigen)
        );
        if( not doReleaseFixed_ or not JsonUtils::fetchValue(parSet.getConfig(), "releaseFixedParametersOnHesse", true) ){
          if( parSet.isEigenParFixed(iEigen) ) {
            _minimizer_->FixVariable(iPar);
          }
        }
      }
    }

  } // parSet

}
