//
// Created by Adrien BLANCHET on 16/12/2021.
//

#include "MinimizerInterface.h"
#include "FitterEngine.h"
#include "JsonUtils.h"
#include "GlobalVariables.h"

#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "Logger.h"

#include "Math/Factory.h"
#include "Math/Minimizer.h"
#include "Math/Functor.h"
#include "TLegend.h"


LoggerInit([]{
  Logger::setUserHeaderStr("[MinimizerInterface]");
});


MinimizerInterface::MinimizerInterface(FitterEngine* owner_): _owner_(owner_){}

void MinimizerInterface::setOwner(FitterEngine* owner_){ _owner_ = owner_; }
void MinimizerInterface::setEnablePostFitErrorEval(bool enablePostFitErrorEval_){ _enablePostFitErrorEval_ = enablePostFitErrorEval_; }

void MinimizerInterface::readConfigImpl(){
  LogInfo << "Reading minimizer config..." << std::endl;

  _minimizerType_ = JsonUtils::fetchValue(_config_, "minimizer", _minimizerType_);
  _minimizerAlgo_ = JsonUtils::fetchValue(_config_, "algorithm", _minimizerAlgo_);

  _useNormalizedFitSpace_ = JsonUtils::fetchValue(_config_, "useNormalizedFitSpace", _useNormalizedFitSpace_);

  _strategy_ = JsonUtils::fetchValue(_config_, "strategy", _strategy_);
  _printLevel_ = JsonUtils::fetchValue(_config_, "print_level", _printLevel_);
  _tolerance_ = JsonUtils::fetchValue(_config_, "tolerance", _tolerance_);
  _maxIterations_ = JsonUtils::fetchValue(_config_, {{"maxIterations"}, {"max_iter"}}, _maxIterations_ );
  _maxFcnCalls_ = JsonUtils::fetchValue(_config_, {{"maxFcnCalls"}, {"max_fcn"}}, _maxFcnCalls_ );

  _enableSimplexBeforeMinimize_ = JsonUtils::fetchValue(_config_, "enableSimplexBeforeMinimize", _enableSimplexBeforeMinimize_);
  _simplexMaxFcnCalls_ = JsonUtils::fetchValue(_config_, "simplexMaxFcnCalls", _simplexMaxFcnCalls_);
  _simplexToleranceLoose_ = JsonUtils::fetchValue(_config_, "simplexToleranceLoose", _simplexToleranceLoose_);

  _errorAlgo_ = JsonUtils::fetchValue(_config_, {{"errorsAlgo"}, {"errors"}}, "Hesse");
  _enablePostFitErrorEval_ = JsonUtils::fetchValue(_config_, "enablePostFitErrorFit", _enablePostFitErrorEval_);
  _restoreStepSizeBeforeHesse_ = JsonUtils::fetchValue(_config_, "restoreStepSizeBeforeHesse", _restoreStepSizeBeforeHesse_);

  _generatedPostFitParBreakdown_ = JsonUtils::fetchValue(_config_, "generatedPostFitParBreakdown", _generatedPostFitParBreakdown_);
  _generatedPostFitEigenBreakdown_ = JsonUtils::fetchValue(_config_, "generatedPostFitEigenBreakdown", _generatedPostFitEigenBreakdown_);

}
void MinimizerInterface::initializeImpl(){
  LogInfo << "Initializing the minimizer..." << std::endl;
  LogThrowIf( _owner_== nullptr, "FitterEngine ptr not set." );

  _minimizer_ = std::unique_ptr<ROOT::Math::Minimizer>(
      ROOT::Math::Factory::CreateMinimizer(_minimizerType_, _minimizerAlgo_)
  );
  LogThrowIf(_minimizer_ == nullptr, "Could not create minimizer: " << _minimizerType_ << "/" << _minimizerAlgo_)
  if( _minimizerAlgo_.empty() ){
    _minimizerAlgo_ = _minimizer_->Options().MinimizerAlgorithm();
    LogWarning << "Using default minimizer algo: " << _minimizerAlgo_ << std::endl;
  }

  LogWarning << "Fetching the effective number of fit parameters..." << std::endl;
  _minimizerFitParameterPtr_.clear();
  for( auto& parSet : _owner_->getPropagator().getParameterSetsList() ){
    for( auto& par : parSet.getEffectiveParameterList() ){
      if( par.isEnabled() and not par.isFixed() ) {
        _minimizerFitParameterPtr_.emplace_back(&par);
      }
    }
  }
  _nbFitParameters_ = int(_minimizerFitParameterPtr_.size());

  LogInfo << "Building functor..." << std::endl;
  _functor_ = std::make_unique<ROOT::Math::Functor>(this, &MinimizerInterface::evalFit, _nbFitParameters_);

  _minimizer_->SetFunction(*_functor_);
  _minimizer_->SetStrategy(_strategy_);
  _minimizer_->SetPrintLevel(_printLevel_);
  _minimizer_->SetTolerance(_tolerance_);
  _minimizer_->SetMaxIterations(_maxIterations_);
  _minimizer_->SetMaxFunctionCalls(_maxFcnCalls_);

  _nbFitBins_ = 0;
  for( auto& sample : _owner_->getPropagator().getFitSampleSet().getFitSampleList() ){
    _nbFitBins_ += int(sample.getBinning().getBinsList().size());
  }

  for( int iFitPar = 0 ; iFitPar < _nbFitParameters_ ; iFitPar++ ){
    auto& fitPar = *_minimizerFitParameterPtr_[iFitPar];

    if( not _useNormalizedFitSpace_ ){
      _minimizer_->SetVariable(iFitPar, fitPar.getFullTitle(),fitPar.getParameterValue(),fitPar.getStepSize());
      if(fitPar.getMinValue() == fitPar.getMinValue()){ _minimizer_->SetVariableLowerLimit(iFitPar, fitPar.getMinValue()); }
      if(fitPar.getMaxValue() == fitPar.getMaxValue()){ _minimizer_->SetVariableUpperLimit(iFitPar, fitPar.getMaxValue()); }
      // Changing the boundaries, change the value/step size?
      _minimizer_->SetVariableValue(iFitPar, fitPar.getParameterValue());
      _minimizer_->SetVariableStepSize(iFitPar, fitPar.getStepSize());
    }
    else{
      _minimizer_->SetVariable( iFitPar,fitPar.getFullTitle(),
                                FitParameterSet::toNormalizedParValue(fitPar.getParameterValue(), fitPar),
                                FitParameterSet::toNormalizedParRange(fitPar.getStepSize(), fitPar)
      );
      if(fitPar.getMinValue() == fitPar.getMinValue()){ _minimizer_->SetVariableLowerLimit(iFitPar, FitParameterSet::toNormalizedParValue(fitPar.getMinValue(), fitPar)); }
      if(fitPar.getMaxValue() == fitPar.getMaxValue()){ _minimizer_->SetVariableUpperLimit(iFitPar, FitParameterSet::toNormalizedParValue(fitPar.getMaxValue(), fitPar)); }
      // Changing the boundaries, change the value/step size?
      _minimizer_->SetVariableValue(iFitPar, FitParameterSet::toNormalizedParValue(fitPar.getParameterValue(), fitPar));
      _minimizer_->SetVariableStepSize(iFitPar, FitParameterSet::toNormalizedParRange(fitPar.getStepSize(), fitPar));
    }
  }

  _convergenceMonitor_.addDisplayedQuantity("VarName");
  _convergenceMonitor_.addDisplayedQuantity("LastAddedValue");
  _convergenceMonitor_.addDisplayedQuantity("SlopePerCall");

  _convergenceMonitor_.getQuantity("VarName").title = "Likelihood";
  _convergenceMonitor_.getQuantity("LastAddedValue").title = "Current Value";
  _convergenceMonitor_.getQuantity("SlopePerCall").title = "Avg. Slope /call";

  _convergenceMonitor_.addVariable("Total");
  _convergenceMonitor_.addVariable("Stat");
  _convergenceMonitor_.addVariable("Syst");
}

bool MinimizerInterface::isFitHasConverged() const {
  return _fitHasConverged_;
}
bool MinimizerInterface::isEnablePostFitErrorEval() const {
  return _enablePostFitErrorEval_;
}
GenericToolbox::VariablesMonitor &MinimizerInterface::getConvergenceMonitor() {
  return _convergenceMonitor_;
}
std::vector<FitParameter *> &MinimizerInterface::getMinimizerFitParameterPtr() {
  return _minimizerFitParameterPtr_;
}
const std::unique_ptr<ROOT::Math::Minimizer> &MinimizerInterface::getMinimizer() const {
  return _minimizer_;
}

void MinimizerInterface::minimize() {
  LogThrowIf(not isInitialized(), "not initialized");

  _chi2HistoryTree_ = std::make_unique<TTree>("chi2History", "chi2History");
  _chi2HistoryTree_->SetDirectory(nullptr);
  _chi2HistoryTree_->Branch("nbFitCalls", &_nbFitCalls_);
  _chi2HistoryTree_->Branch("chi2Total", _owner_->getPropagator().getLlhBufferPtr());
  _chi2HistoryTree_->Branch("chi2Stat", _owner_->getPropagator().getLlhStatBufferPtr());
  _chi2HistoryTree_->Branch("chi2Pulls", _owner_->getPropagator().getLlhPenaltyBufferPtr());

  LogWarning << std::endl << GenericToolbox::addUpDownBars("Summary of the fit parameters:") << std::endl;
  for( const auto& parSet : _owner_->getPropagator().getParameterSetsList() ){

    GenericToolbox::TablePrinter t;
    t.setColTitles({ {"Title"}, {"Starting"}, {"Prior"}, {"StdDev"}, {"Min"}, {"Max"}, {"Status"} });

    auto& parList = parSet.getEffectiveParameterList();
    LogWarning << parSet.getName() << ": " << parList.size() << " parameters" << std::endl;
    if( parList.empty() ) continue;

    for( const auto& par : parList ){
      std::string colorStr;
      std::string statusStr;

      if( not par.isEnabled() ) { statusStr = "Disabled"; colorStr = GenericToolbox::ColorCodes::yellowBackground; }
      else if( par.isFixed() )  { statusStr = "Fixed";    colorStr = GenericToolbox::ColorCodes::redBackground; }
      else                      {
        statusStr = PriorType::PriorTypeEnumNamespace::toString(par.getPriorType(), true) + " Prior";
        if(par.getPriorType()==PriorType::Flat) colorStr = GenericToolbox::ColorCodes::blueBackground;
      }

#ifdef NOCOLOR
      colorStr = "";
#endif

      t.addTableLine({
                         par.getTitle(),
                         std::to_string( par.getParameterValue() ),
                         std::to_string( par.getPriorValue() ),
                         std::to_string( par.getStdDevValue() ),
                         std::to_string( par.getMinValue() ),
                         std::to_string( par.getMaxValue() ),
                         statusStr
                     }, colorStr);
    }

    t.printTable();
  }

  _owner_->getPropagator().updateLlhCache();

  LogWarning << std::endl << GenericToolbox::addUpDownBars("Calling minimize...") << std::endl;
  LogInfo << "Number of defined parameters: " << _minimizer_->NDim() << std::endl
          << "Number of free parameters   : " << _minimizer_->NFree() << std::endl
          << "Number of fixed parameters  : " << _minimizer_->NDim() - _minimizer_->NFree() << std::endl
          << "Number of fit bins : " << _nbFitBins_ << std::endl
          << "Chi2 # DoF : " << _nbFitBins_ - _minimizer_->NFree()
          << std::endl;

  int nbFitCallOffset = _nbFitCalls_;
  LogInfo << "Fit call offset: " << nbFitCallOffset << std::endl;

  if( _enableSimplexBeforeMinimize_ ){
    LogWarning << "Running simplex algo before the minimizer" << std::endl;
    LogThrowIf(_minimizerType_ != "Minuit2", "Can't launch simplex with " << _minimizerType_);

    std::string originalAlgo = _minimizer_->Options().MinimizerAlgorithm();

    _minimizer_->Options().SetMinimizerAlgorithm("Simplex");
    _minimizer_->SetMaxFunctionCalls(_simplexMaxFcnCalls_);
    _minimizer_->SetTolerance( _tolerance_ * _simplexToleranceLoose_ );
    _minimizer_->SetStrategy(0);

    // SIMPLEX
    this->enableFitMonitor();
    _fitHasConverged_ = _minimizer_->Minimize();
    this->disableFitMonitor();

    // Back to original
    _minimizer_->Options().SetMinimizerAlgorithm(originalAlgo.c_str());
    _minimizer_->SetMaxFunctionCalls(_maxFcnCalls_);
    _minimizer_->SetTolerance(_tolerance_);
    _minimizer_->SetStrategy(_strategy_);

    LogInfo << _convergenceMonitor_.generateMonitorString(); // lasting printout
    LogWarning << "Simplex ended after " << _nbFitCalls_ - nbFitCallOffset << " calls." << std::endl;
  }

  this->enableFitMonitor();
  _fitHasConverged_ = _minimizer_->Minimize();
  this->disableFitMonitor();

  int nbMinimizeCalls = _nbFitCalls_ - nbFitCallOffset;

  LogInfo << _convergenceMonitor_.generateMonitorString(); // lasting printout
  LogInfo << "Minimization ended after " << nbMinimizeCalls << " calls." << std::endl;
  if(_minimizerAlgo_ == "Migrad") LogWarning << "Status code: " << this->minuitStatusCodeStr.at(_minimizer_->Status()) << std::endl;
  else LogWarning << "Status code: " << _minimizer_->Status() << std::endl;
  if(_minimizerAlgo_ == "Migrad") LogWarning << "Covariance matrix status code: " << this->covMatrixStatusCodeStr.at(_minimizer_->CovMatrixStatus()) << std::endl;
  else LogWarning << "Covariance matrix status code: " << _minimizer_->CovMatrixStatus() << std::endl;

  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(_owner_->getSaveDir(), "fit"), _chi2HistoryTree_.get());

  if( _fitHasConverged_ ){ LogInfo << "Minimization has converged!" << std::endl; }
  else{ LogError << "Minimization did not converged." << std::endl; }


  LogInfo << "Writing convergence stats..." << std::endl;
  int toyIndex = _owner_->getPropagator().getIThrow();
  int nIterations = int(_minimizer_->NIterations());
  double edmBestFit = _minimizer_->Edm();
  double fitStatus = _minimizer_->Status();
  double covStatus = _minimizer_->CovMatrixStatus();
  double chi2MinFitter = _minimizer_->MinValue();

  auto bestFitStats = std::make_unique<TTree>("bestFitStats", "bestFitStats");
  bestFitStats->SetDirectory(nullptr);
  bestFitStats->Branch("fitConverged", &_fitHasConverged_);
  bestFitStats->Branch("fitStatusCode", &fitStatus);
  bestFitStats->Branch("covStatusCode", &covStatus);
  bestFitStats->Branch("edmBestFit", &edmBestFit);
  bestFitStats->Branch("nIterations", &nIterations);
  bestFitStats->Branch("chi2MinFitter", &chi2MinFitter);
  bestFitStats->Branch("toyIndex", &toyIndex);
  bestFitStats->Branch("nCallsAtBestFit", &_nbFitCalls_);
  bestFitStats->Branch("chi2BestFit", _owner_->getPropagator().getLlhBufferPtr());
  bestFitStats->Branch("chi2StatBestFit", _owner_->getPropagator().getLlhStatBufferPtr());
  bestFitStats->Branch("chi2PullsBestFit", _owner_->getPropagator().getLlhPenaltyBufferPtr());

  std::vector<GenericToolbox::RawDataArray> samplesArrList(_owner_->getPropagator().getFitSampleSet().getFitSampleList().size());
  int iSample{-1};
  for( auto& sample : _owner_->getPropagator().getFitSampleSet().getFitSampleList() ){
    if( not sample.isEnabled() ) continue;

    std::vector<std::string> leavesDict;
    iSample++;

    leavesDict.emplace_back("llhSample/D");
    samplesArrList[iSample].writeRawData(_owner_->getPropagator().getFitSampleSet().getJointProbabilityFct()->eval(sample));

    int nBins = int(sample.getBinning().getBinsList().size());
    for( int iBin = 1 ; iBin <= nBins ; iBin++ ){
      leavesDict.emplace_back("llhSample_bin" + std::to_string(iBin) + "/D");
      samplesArrList[iSample].writeRawData(_owner_->getPropagator().getFitSampleSet().getJointProbabilityFct()->eval(sample, iBin));
    }

    samplesArrList[iSample].lockArraySize();


    std::string cleanBranchName{sample.getName()};
    GenericToolbox::replaceSubstringInsideInputString(cleanBranchName, " ", "_");
    GenericToolbox::replaceSubstringInsideInputString(cleanBranchName, "-", "_");
    GenericToolbox::replaceSubstringInsideInputString(cleanBranchName, "(", "");
    GenericToolbox::replaceSubstringInsideInputString(cleanBranchName, ")", "");

    bestFitStats->Branch(
        cleanBranchName.c_str(),
        &samplesArrList[iSample].getRawDataArray()[0],
        GenericToolbox::joinVectorString(leavesDict, ":").c_str()
    );
  }

  std::vector<GenericToolbox::RawDataArray> parameterSetArrList(_owner_->getPropagator().getParameterSetsList().size());
  int iParSet{-1};
  for( auto& parSet : _owner_->getPropagator().getParameterSetsList() ){
    if( not parSet.isEnabled() ) continue;

    std::vector<std::string> leavesDict;
    iParSet++;

    leavesDict.emplace_back("llhPenalty/D");
    parameterSetArrList[iParSet].writeRawData(parSet.getPenaltyChi2());

    for( auto& par : parSet.getParameterList() ){
      leavesDict.emplace_back(GenericToolbox::replaceSubstringInString(par.getTitle(), " ", "_") + "/D");
      parameterSetArrList[iParSet].writeRawData(par.getParameterValue());
    }

    std::string cleanBranchName{parSet.getName()};
    GenericToolbox::replaceSubstringInsideInputString(cleanBranchName, " ", "_");
    GenericToolbox::replaceSubstringInsideInputString(cleanBranchName, "-", "_");
    GenericToolbox::replaceSubstringInsideInputString(cleanBranchName, "(", "");
    GenericToolbox::replaceSubstringInsideInputString(cleanBranchName, ")", "");

    bestFitStats->Branch(
        cleanBranchName.c_str(),
        &parameterSetArrList[iParSet].getRawDataArray()[0],
        GenericToolbox::joinVectorString(leavesDict, ":").c_str()
    );
  }

  bestFitStats->Fill();
  GenericToolbox::mkdirTFile(_owner_->getSaveDir(), "postFit")->WriteObject(bestFitStats.get(), bestFitStats->GetName());

  LogInfo << "Writing " << _minimizerType_ << "/" << _minimizerAlgo_ << " post-fit errors" << std::endl;
  this->writePostFitData(GenericToolbox::mkdirTFile(_owner_->getSaveDir(), "postFit/" + _minimizerAlgo_));
  GenericToolbox::triggerTFileWrite(GenericToolbox::mkdirTFile(_owner_->getSaveDir(), "postFit/" + _minimizerAlgo_));
}
void MinimizerInterface::calcErrors(){
  LogThrowIf(not isInitialized(), "not initialized");

  int nbFitCallOffset = _nbFitCalls_;

  LogWarning << std::endl << GenericToolbox::addUpDownBars("Calling HESSE...") << std::endl;
  LogInfo << "Number of defined parameters: " << _minimizer_->NDim() << std::endl
          << "Number of free parameters   : " << _minimizer_->NFree() << std::endl
          << "Number of fixed parameters  : " << _minimizer_->NDim() - _minimizer_->NFree() << std::endl
          << "Number of fit bins : " << _nbFitBins_ << std::endl
          << "Chi2 # DoF : " << _nbFitBins_ - _minimizer_->NFree() << std::endl
          << "Fit call offset: " << nbFitCallOffset << std::endl;

  if     ( _errorAlgo_ == "Minos" ){
    LogWarning << std::endl << GenericToolbox::addUpDownBars("Calling MINOS...") << std::endl;

    double errLow, errHigh;
    _minimizer_->SetPrintLevel(0);

    for( int iFitPar = 0 ; iFitPar < _minimizer_->NDim() ; iFitPar++ ){
      LogInfo << "Evaluating: " << _minimizer_->VariableName(iFitPar) << "..." << std::endl;

      this->enableFitMonitor();
      bool isOk = _minimizer_->GetMinosError(iFitPar, errLow, errHigh);
      this->disableFitMonitor();

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

    // Put back at minimum
    for( int iFitPar = 0 ; iFitPar < _minimizer_->NDim() ; iFitPar++ ){
      _minimizerFitParameterPtr_[iFitPar]->setParameterValue(_minimizer_->X()[iFitPar]);
    }
  } // Minos
  else if( _errorAlgo_ == "Hesse" ){

    if( _restoreStepSizeBeforeHesse_ ){
      LogWarning << "Restoring step size before HESSE..." << std::endl;
      for( int iFitPar = 0 ; iFitPar < _minimizer_->NDim() ; iFitPar++ ){
        auto& par = *_minimizerFitParameterPtr_[iFitPar];
        if(not _useNormalizedFitSpace_){ _minimizer_->SetVariableStepSize(iFitPar, par.getStepSize()); }
        else{ _minimizer_->SetVariableStepSize(iFitPar, FitParameterSet::toNormalizedParRange(par.getStepSize(), par)); } // should be 1
      }
    }

    this->enableFitMonitor();
    _fitHasConverged_ = _minimizer_->Hesse();
    this->disableFitMonitor();

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
    this->writePostFitData(GenericToolbox::mkdirTFile(_owner_->getSaveDir(), "postFit/Hesse"));
    GenericToolbox::triggerTFileWrite(GenericToolbox::mkdirTFile(_owner_->getSaveDir(), "postFit/Hesse"));
  }
  else{
    LogError << GET_VAR_NAME_VALUE(_errorAlgo_) << " not implemented." << std::endl;
  }
}

double MinimizerInterface::evalFit(const double* parArray_){
  LogThrowIf(not isInitialized(), "not initialized");
  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);

  if(_nbFitCalls_ != 0){
    _outEvalFitAvgTimer_.counts++ ; _outEvalFitAvgTimer_.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("out_evalFit");
  }
  _nbFitCalls_++;

  // Update fit parameter values:
  int iFitPar{0};
  for( auto* par : _minimizerFitParameterPtr_ ){
    if( _useNormalizedFitSpace_ ) par->setParameterValue(FitParameterSet::toRealParValue(parArray_[iFitPar++], *par));
    else par->setParameterValue(parArray_[iFitPar++]);
  }

  // Compute the Chi2
  _owner_->getPropagator().updateLlhCache();

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
    ss << std::endl << __METHOD_NAME__ << ": call #" << _nbFitCalls_;
    ss << std::endl << "Target EDM: " << 0.001*_minimizer_->Tolerance()*2;
    ss << std::endl << "Current RAM usage: " << GenericToolbox::parseSizeUnits(double(GenericToolbox::getProcessMemoryUsage()));
    double cpuPercent = GenericToolbox::getCpuUsageByProcess();
    ss << std::endl << "Current CPU usage: " << cpuPercent << "% (" << cpuPercent/GlobalVariables::getNbThreads() << "% efficiency)";
    ss << std::endl << "Avg " << GUNDAM_CHI2 << " computation time: " << _evalFitAvgTimer_;
    ss << std::endl << GUNDAM_CHI2 << "/dof: " << _owner_->getPropagator().getLlhBuffer()/double(_nbFitBins_ - _minimizer_->NFree());
    ss << std::endl;
#ifndef GUNDAM_BATCH
    ss << "├─";
#endif
    ss << " Current speed:                 " << (double)_itSpeed_.counts/(double)_itSpeed_.cumulated * 1E6 << " it/s";
    ss << std::endl;
#ifndef GUNDAM_BATCH
    ss << "├─";
#endif
    ss << " Avg time for " << _minimizerType_ << "/" << _minimizerAlgo_ << ":   " << _outEvalFitAvgTimer_;
    ss << std::endl;
#ifndef GUNDAM_BATCH
    ss << "├─";
#endif
    ss << " Avg time to propagate weights: " << _owner_->getPropagator().weightProp;
    ss << std::endl;
#ifndef GUNDAM_BATCH
    ss << "├─";
#endif
    ss << " Avg time to fill histograms:   " << _owner_->getPropagator().fillProp;
    _convergenceMonitor_.setHeaderString(ss.str());
    _convergenceMonitor_.getVariable("Total").addQuantity(_owner_->getPropagator().getLlhBuffer());
    _convergenceMonitor_.getVariable("Stat").addQuantity(_owner_->getPropagator().getLlhStatBuffer());
    _convergenceMonitor_.getVariable("Syst").addQuantity(_owner_->getPropagator().getLlhPenaltyBuffer());

    if( _nbFitCalls_ == 1 ){
      // don't erase these lines
      LogInfo << _convergenceMonitor_.generateMonitorString();
    }
    else{
      LogInfo << _convergenceMonitor_.generateMonitorString(true , true);
    }

    _itSpeed_.counts = _nbFitCalls_;
  }

  // Fill History
  _chi2HistoryTree_->Fill();
//  _chi2History_["Total"].emplace_back(_owner_->getPropagator().getLlhBuffer());
//  _chi2History_["Stat"].emplace_back(_chi2StatBuffer_);
//  _chi2History_["Syst"].emplace_back(_chi2PullsBuffer_);

  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("out_evalFit");
  return _owner_->getPropagator().getLlhBuffer();
}
void MinimizerInterface::writePostFitData(TDirectory* saveDir_) {
  LogInfo << __METHOD_NAME__ << std::endl;
  LogThrowIf(not isInitialized(), "not initialized");
  LogThrowIf(saveDir_==nullptr, "Save dir not specified");

  LogInfo << "Extracting post-fit covariance matrix" << std::endl;
  auto* matricesDir = GenericToolbox::mkdirTFile(saveDir_, "hessian");

  TMatrixDSym postfitCovarianceMatrix(int(_minimizer_->NDim()));
  _minimizer_->GetCovMatrix(postfitCovarianceMatrix.GetMatrixArray());

  std::function<void(TDirectory*)> decomposeCovarianceMatrixFct = [&](TDirectory* outDir_){

    std::function<void(TH1*)> applyLooks = [&](TH1* hist_){
      if(hist_->GetDimension() == 2){
        hist_->SetDrawOption("COLZ");
        GenericToolbox::fixTH2display((TH2*) hist_);
        hist_->GetYaxis()->SetLabelSize(0.02);
      }
      hist_->GetXaxis()->SetLabelSize(0.02);
    };
    std::function<void(TH1*)> applyBinLabels = [&](TH1* hist_){
      for(int iPar = 0 ; iPar < _minimizer_->NDim() ; iPar++ ){
        hist_->GetXaxis()->SetBinLabel(iPar+1, _minimizer_->VariableName(iPar).c_str());
        if(hist_->GetDimension() >= 2) hist_->GetYaxis()->SetBinLabel(iPar+1, _minimizer_->VariableName(iPar).c_str());
      }
      applyLooks(hist_);
    };

    {
      LogInfo << "Writing post-fit matrices" << std::endl;
      auto postFitCovarianceTH2D = std::unique_ptr<TH2D>(GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) &postfitCovarianceMatrix) );
      applyBinLabels(postFitCovarianceTH2D.get());
      GenericToolbox::writeInTFile(outDir_, postFitCovarianceTH2D.get(), "postfitCovariance");

      auto postfitCorrelationMatrix = std::unique_ptr<TMatrixD>(GenericToolbox::convertToCorrelationMatrix((TMatrixD*) &postfitCovarianceMatrix));
      auto postfitCorrelationTH2D = std::unique_ptr<TH2D>(GenericToolbox::convertTMatrixDtoTH2D(postfitCorrelationMatrix.get()));
      applyBinLabels(postfitCorrelationTH2D.get());
      postfitCorrelationTH2D->GetZaxis()->SetRangeUser(-1,1);
      GenericToolbox::writeInTFile(outDir_, postfitCorrelationTH2D.get(), "postfitCorrelation");
    }

    // Fitter covariance matrix decomposition
    {
      LogInfo << "Eigen decomposition of the post-fit covariance matrix" << std::endl;
      TMatrixDSymEigen decompCovMatrix(postfitCovarianceMatrix);

      auto eigenVectors = std::unique_ptr<TH2D>( GenericToolbox::convertTMatrixDtoTH2D(&decompCovMatrix.GetEigenVectors()) );
      applyBinLabels(eigenVectors.get());
      GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(outDir_, "eigenDecomposition"), eigenVectors.get(), "eigenVectors");

      auto eigenValues = std::unique_ptr<TH1D>( GenericToolbox::convertTVectorDtoTH1D(&decompCovMatrix.GetEigenValues()) );
      applyBinLabels(eigenValues.get());
      GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(outDir_, "eigenDecomposition"), eigenValues.get(), "eigenValues");

      double conditioning = decompCovMatrix.GetEigenValues().Min() / decompCovMatrix.GetEigenValues().Max();
      LogWarning << "Post-fit error conditioning is: " << conditioning << std::endl;

      LogInfo << "Reconstructing postfit hessian matrix..." << std::endl;
      auto eigenValuesInv = TVectorD(decompCovMatrix.GetEigenValues());
      for( int iEigen = 0 ; iEigen < eigenValuesInv.GetNrows() ; iEigen++ ){ eigenValuesInv[iEigen] = 1./eigenValuesInv[iEigen]; }
      auto diagonalMatrixInv = std::unique_ptr<TMatrixD>( GenericToolbox::makeDiagonalMatrix(&eigenValuesInv) );
      auto invEigVectors = TMatrixD(decompCovMatrix.GetEigenVectors());
      invEigVectors.T();

      TMatrixD hessianMatrix(int(_minimizer_->NDim()), int(_minimizer_->NDim())); hessianMatrix.Zero();
      hessianMatrix += decompCovMatrix.GetEigenVectors();
      hessianMatrix *= (*diagonalMatrixInv);
      hessianMatrix *= invEigVectors;

      TH2D* postfitHessianTH2D = GenericToolbox::convertTMatrixDtoTH2D(&hessianMatrix);
      applyBinLabels(postfitHessianTH2D);
      GenericToolbox::writeInTFile(outDir_, postfitHessianTH2D, "postfitHessian");

      if( _generatedPostFitEigenBreakdown_ ){
        LogInfo << "Eigen breakdown..." << std::endl;
        TH1D eigenBreakdownHist("eigenBreakdownHist", "eigenBreakdownHist",
                                int(_minimizer_->NDim()), -0.5, int(_minimizer_->NDim()) - 0.5);
        std::vector<TH1D> eigenBreakdownAccum(decompCovMatrix.GetEigenValues().GetNrows(), eigenBreakdownHist);
        TH1D* lastAccumHist{nullptr};
        std::string progressTitle = LogWarning.getPrefixString() + "Accumulating eigen components...";
        for (int iEigen = decompCovMatrix.GetEigenValues().GetNrows() - 1; iEigen >= 0; iEigen--) {
          GenericToolbox::displayProgressBar(decompCovMatrix.GetEigenValues().GetNrows() - iEigen, decompCovMatrix.GetEigenValues().GetNrows(), progressTitle);
          // iEigen = 0 -> biggest error contribution
          // Drawing in the back -> iEigen = 0 should be last in the accum plot
          if( lastAccumHist != nullptr ) eigenBreakdownAccum[iEigen] = *lastAccumHist;
          else eigenBreakdownAccum[iEigen] = eigenBreakdownHist;
          lastAccumHist = &eigenBreakdownAccum[iEigen];

          eigenBreakdownHist.SetTitle(Form("Parameter breakdown for eigen #%i = %f", iEigen,
                                           decompCovMatrix.GetEigenValues()[iEigen]));
          eigenBreakdownHist.SetLineColor(GenericToolbox::defaultColorWheel[iEigen%int(GenericToolbox::defaultColorWheel.size())]);
          eigenBreakdownHist.SetLabelSize(0.02);
          for (int iPar = int(_minimizer_->NDim())-1; iPar >= 0; iPar--) {
            eigenBreakdownHist.SetBinContent(iPar + 1,
                                             decompCovMatrix.GetEigenVectors()[iPar][iEigen] *
                                             decompCovMatrix.GetEigenVectors()[iPar][iEigen] *
                                             decompCovMatrix.GetEigenValues()[iEigen]
            );
          }
          applyBinLabels(&eigenBreakdownHist);
          applyBinLabels(&eigenBreakdownAccum[iEigen]);

          GenericToolbox::writeInTFile(
              GenericToolbox::mkdirTFile(outDir_, "eigenDecomposition/eigenBreakdown"),
              &eigenBreakdownHist, Form("eigen#%i", iEigen));

          eigenBreakdownAccum[iEigen].Add(&eigenBreakdownHist);
          eigenBreakdownAccum[iEigen].SetLabelSize(0.02);
          eigenBreakdownAccum[iEigen].SetLineColor(kBlack);
          eigenBreakdownAccum[iEigen].SetFillColor(GenericToolbox::defaultColorWheel[iEigen%int(GenericToolbox::defaultColorWheel.size())]);

          int cycle = iEigen/int(GenericToolbox::defaultColorWheel.size());
          if( cycle > 0 ) eigenBreakdownAccum[iEigen].SetFillStyle( short(3044 + 100 * (cycle%10)) );
          else eigenBreakdownAccum[iEigen].SetFillStyle(1001);
        }

        TCanvas accumPlot("accumPlot", "accumPlot", 1280, 720);
        TLegend l(0.15, 0.4, 0.3, 0.85);
        bool isFirst{true};
        for (int iEigen = 0; iEigen < int(eigenBreakdownAccum.size()); iEigen++) {
          if( iEigen < GenericToolbox::defaultColorWheel.size() ){
            l.AddEntry(&eigenBreakdownAccum[iEigen], Form("Eigen #%i = %f", iEigen, decompCovMatrix.GetEigenValues()[iEigen]));
          }
          accumPlot.cd();
          if( isFirst ){
            eigenBreakdownAccum[iEigen].SetTitle("Hessian eigen composition of post-fit errors");
            eigenBreakdownAccum[iEigen].GetYaxis()->SetRangeUser(0, eigenBreakdownAccum[iEigen].GetMaximum()*1.2);
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
        GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(outDir_, "eigenDecomposition"), &accumPlot, "eigenBreakdown");
      }

      if( _generatedPostFitParBreakdown_ ){
        LogInfo << "Parameters breakdown..." << std::endl;
        TH1D parBreakdownHist("parBreakdownHist", "parBreakdownHist",
                              decompCovMatrix.GetEigenValues().GetNrows(), -0.5,
                              decompCovMatrix.GetEigenValues().GetNrows() - 0.5);
        std::vector<TH1D> parBreakdownAccum(_minimizer_->NDim());
        TH1D* lastAccumHist{nullptr};
        for (int iPar = int(_minimizer_->NDim())-1; iPar >= 0; iPar--){

          if( lastAccumHist != nullptr ) parBreakdownAccum[iPar] = *lastAccumHist;
          else parBreakdownAccum[iPar] = parBreakdownHist;
          lastAccumHist = &parBreakdownAccum[iPar];

          parBreakdownHist.SetLineColor(GenericToolbox::defaultColorWheel[iPar%int(GenericToolbox::defaultColorWheel.size())]);

          parBreakdownHist.SetTitle(Form("Eigen breakdown for parameter #%i: %s", iPar, _minimizer_->VariableName(iPar).c_str()));
          for (int iEigen = decompCovMatrix.GetEigenValues().GetNrows() - 1; iEigen >= 0; iEigen--){
            parBreakdownHist.SetBinContent(
                iPar+1,
                decompCovMatrix.GetEigenVectors()[iPar][iEigen]
                * decompCovMatrix.GetEigenVectors()[iPar][iEigen]
                * decompCovMatrix.GetEigenValues()[iEigen]
            );
          }
          GenericToolbox::writeInTFile(
              GenericToolbox::mkdirTFile(outDir_, "eigenDecomposition/parBreakdown"),
              &parBreakdownHist, Form("par#%i", iPar)
              );

          parBreakdownAccum[iPar].Add(&parBreakdownHist);
          parBreakdownAccum[iPar].SetLabelSize(0.02);
          parBreakdownAccum[iPar].SetLineColor(kBlack);
          parBreakdownAccum[iPar].SetFillColor(GenericToolbox::defaultColorWheel[iPar%int(GenericToolbox::defaultColorWheel.size())]);
        }
        TCanvas accumPlot("accumParPlot", "accumParPlot", 1280, 720);
        bool isFirst{true};
        for (auto & parHist : parBreakdownAccum) {
          accumPlot.cd();
          isFirst ? parHist.Draw("HIST") : parHist.Draw("HIST SAME");
          isFirst = false;
        }
        GenericToolbox::writeInTFile(
            GenericToolbox::mkdirTFile(outDir_, "eigenDecomposition"),
            &accumPlot, "parBreakdown"
            );
      }

    }

    // Post-fit covariance matrix in original phase space
    {
      std::function<void(TH1*, const std::vector<std::string>&)> applyBinLabelsOrig = [&](TH1* hist_, const std::vector<std::string>& labels_){
        for(int iPar = 0 ; iPar < int(labels_.size()) ; iPar++ ){
          hist_->GetXaxis()->SetBinLabel(iPar+1, labels_[iPar].c_str());
          if(hist_->GetDimension() >= 2) hist_->GetYaxis()->SetBinLabel(iPar+1, labels_[iPar].c_str());
        }
        applyLooks(hist_);
      };

      int nGlobalPars{0};
      for( const auto& parSet : _owner_->getPropagator().getParameterSetsList() ){ if( parSet.isEnabled() ) nGlobalPars += int(parSet.getNbParameters()); }

      // Reconstruct the global passage matrix
      std::vector<std::string> parameterLabels(nGlobalPars);
      auto globalPassageMatrix = std::make_unique<TMatrixD>(nGlobalPars, nGlobalPars);
      for(int i = 0 ; i < nGlobalPars; i++ ){ (*globalPassageMatrix)[i][i] = 1; }
      int blocOffset{0};
      for( const auto& parSet : _owner_->getPropagator().getParameterSetsList() ){
        if( not parSet.isEnabled() ) continue;

        auto* parList = &parSet.getParameterList(); // we want the original names
        for( auto& par : *parList ){ parameterLabels[blocOffset + par.getParameterIndex()] = par.getFullTitle(); }

        parList = &parSet.getEffectiveParameterList();
        if( parSet.isUseEigenDecompInFit() ){
          int iParIdx{0};
          for( auto& iPar : *parList ){
            int jParIdx{0};
            for( auto& jPar : *parList ){
              (*globalPassageMatrix)[blocOffset + iPar.getParameterIndex()][blocOffset + jPar.getParameterIndex()] = (*parSet.getEigenVectors())[iParIdx][jParIdx];
              jParIdx++;
            }
            iParIdx++;
          }
        }

        blocOffset += int(parList->size());
      }

      // Reconstruct the global cov matrix (including eigen decomp parameters)
      auto unstrippedCovMatrix = std::make_unique<TMatrixD>(nGlobalPars, nGlobalPars);
      int iOffset{0};
      for( const auto& iParSet : _owner_->getPropagator().getParameterSetsList() ){
        if( not iParSet.isEnabled() ) continue;

        auto* iParList = &iParSet.getEffectiveParameterList();
        for( auto& iPar : *iParList ){
          int iMinimizerIndex = GenericToolbox::findElementIndex((FitParameter*) &iPar, _minimizerFitParameterPtr_);

          int jOffset{0};
          for( const auto& jParSet : _owner_->getPropagator().getParameterSetsList() ){
            if( not jParSet.isEnabled() ) continue;

            auto* jParList = &jParSet.getEffectiveParameterList();
            for( auto& jPar : *jParList ){
              int jMinimizerIndex = GenericToolbox::findElementIndex((FitParameter*) &jPar, _minimizerFitParameterPtr_);

              if( iMinimizerIndex != -1 and jMinimizerIndex != -1 ){
                // Use the fit-constrained value
                (*unstrippedCovMatrix)[iOffset + iPar.getParameterIndex()][jOffset + jPar.getParameterIndex()] = postfitCovarianceMatrix[iMinimizerIndex][jMinimizerIndex];
              }
              else{
                // Inherit from the prior in eigen -> only diagonal are non 0
                if( &iParSet == &jParSet and iParSet.isUseEigenDecompInFit() ){
                  if( iPar.getParameterIndex() == jPar.getParameterIndex() ){
                    (*unstrippedCovMatrix)[iOffset + iPar.getParameterIndex()][jOffset + jPar.getParameterIndex()] = iPar.getStdDevValue()*iPar.getStdDevValue();
                  }
                }
              }
            }
            jOffset += int(jParList->size());
          }
        }
        iOffset += int(iParList->size());
      }

      // Get the invert passage matrix
      auto globalPassageMatrixInv = std::make_unique<TMatrixD>(TMatrixD::kTransposed, *globalPassageMatrix);

      auto originalStrippedCovMatrix = std::make_unique<TMatrixD>(unstrippedCovMatrix->GetNrows(), unstrippedCovMatrix->GetNcols());
      (*originalStrippedCovMatrix) =  (*globalPassageMatrix);
      (*originalStrippedCovMatrix) *= (*unstrippedCovMatrix);
      (*originalStrippedCovMatrix) *= (*globalPassageMatrixInv);

      int nParNonFixed{0};
      for( int i = 0 ; i < originalStrippedCovMatrix->GetNrows() ; i++ ){
        if( (*originalStrippedCovMatrix)[i][i] != 0 ) nParNonFixed++;
      }
      std::vector<std::string> parameterNonFixedLabels(nParNonFixed);
      auto originalCovMatrix = std::make_unique<TMatrixD>(nParNonFixed, nParNonFixed);
      int iStrip{0};
      for( int i = 0 ; i < originalStrippedCovMatrix->GetNrows() ; i++ ){
        if( (*originalStrippedCovMatrix)[i][i] == 0 ) continue;

        parameterNonFixedLabels[iStrip] = parameterLabels[i];

        int jStrip{0};
        for( int j = 0 ; j < originalStrippedCovMatrix->GetNrows() ; j++ ){
          if( (*originalStrippedCovMatrix)[j][j] == 0 ) continue;

          (*originalCovMatrix)[iStrip][jStrip] = (*originalStrippedCovMatrix)[i][j];
          jStrip++;
        }
        iStrip++;
      }

      TH2D* postfitCovarianceOriginalTH2D = GenericToolbox::convertTMatrixDtoTH2D(originalCovMatrix.get());
      applyBinLabelsOrig(postfitCovarianceOriginalTH2D, parameterNonFixedLabels);
      GenericToolbox::writeInTFile(outDir_, postfitCovarianceOriginalTH2D, "postfitCovarianceOriginal");

      TH2D* postfitCorrelationOriginalTH2D = GenericToolbox::convertTMatrixDtoTH2D(GenericToolbox::convertToCorrelationMatrix(originalCovMatrix.get()));
      applyBinLabelsOrig(postfitCorrelationOriginalTH2D, parameterNonFixedLabels);
      postfitCorrelationOriginalTH2D->GetZaxis()->SetRangeUser(-1,1);
      GenericToolbox::writeInTFile(outDir_, postfitCorrelationOriginalTH2D, "postfitCorrelationOriginal");
    }

  };

  if( _useNormalizedFitSpace_ ){
    LogInfo << "Writing normalized decomposition of the output matrix..." << std::endl;
    decomposeCovarianceMatrixFct(GenericToolbox::mkdirTFile(matricesDir, "normalizedFitSpace"));

    // Rescale the post-fit values:
    for(int iRow = 0 ; iRow < postfitCovarianceMatrix.GetNrows() ; iRow++ ){
      for(int iCol = 0 ; iCol < postfitCovarianceMatrix.GetNcols() ; iCol++ ){
        postfitCovarianceMatrix[iRow][iCol] *= (_minimizerFitParameterPtr_[iRow]->getStdDevValue()) * (_minimizerFitParameterPtr_[iCol]->getStdDevValue());
      }
    }

  }

  LogInfo << "Writing decomposition of the output matrix..." << std::endl;
  decomposeCovarianceMatrixFct(matricesDir);

  LogInfo << "Fitter covariance matrix is " << postfitCovarianceMatrix.GetNrows() << "x" << postfitCovarianceMatrix.GetNcols() << std::endl;
  auto* errorDir = GenericToolbox::mkdirTFile(saveDir_, "errors");

  auto savePostFitObjFct =
      [&](const FitParameterSet& parSet_, const std::vector<FitParameter>& parList_, TMatrixD* covMatrix_, TDirectory* saveSubdir_){

        auto* covMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) covMatrix_, Form("Covariance_%s", parSet_.getName().c_str()));
        auto* corMatrix = GenericToolbox::convertToCorrelationMatrix((TMatrixD*) covMatrix_);
        auto* corMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D(corMatrix, Form("Correlation_%s", parSet_.getName().c_str()));

        size_t maxLabelLength{0};
        for( const auto& par : parList_ ){
          maxLabelLength = std::max(maxLabelLength, par.getTitle().size());
          covMatrixTH2D->GetXaxis()->SetBinLabel(1+par.getParameterIndex(), par.getTitle().c_str());
          covMatrixTH2D->GetYaxis()->SetBinLabel(1+par.getParameterIndex(), par.getTitle().c_str());
          corMatrixTH2D->GetXaxis()->SetBinLabel(1+par.getParameterIndex(), par.getTitle().c_str());
          corMatrixTH2D->GetYaxis()->SetBinLabel(1+par.getParameterIndex(), par.getTitle().c_str());
        }

        auto corMatrixCanvas = std::make_unique<TCanvas>("host", "host", 1024, 1024);
        corMatrixCanvas->cd();
        corMatrixTH2D->GetXaxis()->SetLabelSize(0.025);
        corMatrixTH2D->GetXaxis()->LabelsOption("v");
        corMatrixTH2D->GetXaxis()->SetTitle("");
        corMatrixTH2D->GetYaxis()->SetLabelSize(0.025);
        corMatrixTH2D->GetYaxis()->SetTitle("");
        corMatrixTH2D->GetZaxis()->SetRangeUser(-1,1);
        corMatrixTH2D->GetZaxis()->SetTitle("Correlation");
        corMatrixTH2D->GetZaxis()->SetTitleOffset(1.1);
        corMatrixTH2D->SetTitle(Form("Post-fit correlation matrix for %s", parSet_.getName().c_str()));
        corMatrixTH2D->Draw("COLZ");

        GenericToolbox::fixTH2display(corMatrixTH2D);
        auto* pal = (TPaletteAxis*) corMatrixTH2D->GetListOfFunctions()->FindObject("palette");
        // TPaletteAxis* pal = (TPaletteAxis*) histogram_->GetListOfFunctions()->At(0);
        if(pal != nullptr){
          pal->SetY1NDC(0.15);
          pal->SetTitleOffset(2);
          pal->Draw();
        }
        gPad->SetLeftMargin(float(0.1*(1. + double(maxLabelLength)/20.)));
        gPad->SetBottomMargin(float(0.1*(1. + double(maxLabelLength)/15.)));

        corMatrixTH2D->Draw("COLZ");

        GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(saveSubdir_, "matrices"), covMatrix_, "Covariance");
        GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(saveSubdir_, "matrices"), covMatrixTH2D, "Covariance");
        GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(saveSubdir_, "matrices"), corMatrix, "Correlation");
        GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(saveSubdir_, "matrices"), corMatrixTH2D, "Correlation");
        GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(saveSubdir_, "matrices"), corMatrix, "Correlation");
        GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(saveSubdir_, "matrices"), corMatrixCanvas.get(), "Correlation");

        // Table printout
        std::vector<std::vector<std::string>> tableLines;
        tableLines.emplace_back(std::vector<std::string>{
            "Parameter"
            ,"Prior Value"
            ,"Fit Value"
            ,"Prior Err"
            ,"Fit Err"
            ,"Constraint"
        });
        for( const auto& par : parList_ ){
          if( par.isEnabled() and not par.isFixed() ){
            double priorFraction = TMath::Sqrt((*covMatrix_)[par.getParameterIndex()][par.getParameterIndex()]) / par.getStdDevValue();
            std::stringstream ss;
#ifndef NOCOLOR
            std::string red(GenericToolbox::ColorCodes::redBackground);
            std::string ylw(GenericToolbox::ColorCodes::yellowBackground);
            std::string rst(GenericToolbox::ColorCodes::resetColor);
#else
            std::string red;
        std::string ylw;
        std::string rst;
#endif

            if( priorFraction < 1E-2 ) ss << ylw;
            if( priorFraction > 1 ) ss << red;
            std::vector<std::string> lineValues(tableLines[0].size());
            int valIndex{0};
            lineValues[valIndex++] = par.getFullTitle();
            lineValues[valIndex++] = std::to_string( par.getPriorValue() );
            lineValues[valIndex++] = std::to_string( par.getParameterValue() );
            lineValues[valIndex++] = std::to_string( par.getStdDevValue() );
            lineValues[valIndex++] = std::to_string( TMath::Sqrt((*covMatrix_)[par.getParameterIndex()][par.getParameterIndex()]) );

            std::string colorStr;
            if( par.isFree() ){
              lineValues[valIndex++] = "Unconstrained";
              colorStr = GenericToolbox::ColorCodes::blueBackground;
            }
            else{
              lineValues[valIndex++] = std::to_string( priorFraction*100 ) + " \%";
              if( priorFraction > 1 ){ colorStr = GenericToolbox::ColorCodes::redBackground; }
            }

#ifndef NOCOLOR
            if( not colorStr.empty() ){
              for( auto& line : lineValues ){
                if(not line.empty()){
                  line.insert(0, colorStr);
                  line += GenericToolbox::ColorCodes::resetColor;
                }
              }
            }
#endif

            tableLines.emplace_back(lineValues);
          }
        }
        GenericToolbox::TablePrinter t;
        t.fillTable(tableLines);
        t.printTable();

        // Parameters plots
        auto makePrePostFitCompPlot = [&](TDirectory* saveDir_, bool isNorm_){
          size_t longestTitleSize{0};

          auto postFitErrorHist   = std::make_unique<TH1D>("postFitErrors", "Post-fit Errors", parSet_.getNbParameters(), 0, parSet_.getNbParameters());
          auto preFitErrorHist    = std::make_unique<TH1D>("preFitErrors", "Pre-fit Errors", parSet_.getNbParameters(), 0, parSet_.getNbParameters());
          auto toyParametersLine  = std::make_unique<TH1D>("toyParametersLine", "toyParametersLine", parSet_.getNbParameters(), 0, parSet_.getNbParameters());

          std::vector<TBox> freeParBoxes;
          std::vector<TBox> fixedParBoxes;

          auto legend = std::make_unique<TLegend>(0.6, 0.79, 0.89, 0.89);
          legend->AddEntry(preFitErrorHist.get(),"Pre-fit values","fl");
          legend->AddEntry(postFitErrorHist.get(),"Post-fit values","ep");

          for( const auto& par : parList_ ){
            longestTitleSize = std::max(longestTitleSize, par.getTitle().size());

            postFitErrorHist->GetXaxis()->SetBinLabel(1 + par.getParameterIndex(), par.getTitle().c_str());
            preFitErrorHist->GetXaxis()->SetBinLabel(1 + par.getParameterIndex(), par.getTitle().c_str());

            if(not isNorm_){
              postFitErrorHist->SetBinContent( 1 + par.getParameterIndex(), par.getParameterValue());
              postFitErrorHist->SetBinError( 1 + par.getParameterIndex(), TMath::Sqrt((*covMatrix_)[par.getParameterIndex()][par.getParameterIndex()]));
              preFitErrorHist->SetBinContent( 1 + par.getParameterIndex(), par.getPriorValue() );

              if( par.isEnabled() and not par.isFixed() and not par.isFree() ){
                preFitErrorHist->SetBinError( 1 + par.getParameterIndex(), par.getStdDevValue() );
              }
            }
            else{
              postFitErrorHist->SetBinContent(
                  1 + par.getParameterIndex(),
                  FitParameterSet::toNormalizedParValue(par.getParameterValue(), par)
              );
              preFitErrorHist->SetBinContent( 1 + par.getParameterIndex(), 0 );

              postFitErrorHist->SetBinError(
                  1 + par.getParameterIndex(),
                  FitParameterSet::toNormalizedParRange(
                      TMath::Sqrt((*covMatrix_)[par.getParameterIndex()][par.getParameterIndex()]), par
                  )
              );
              if( par.isEnabled() and not par.isFixed() and not par.isFree() ){
                preFitErrorHist->SetBinError( 1 + par.getParameterIndex(), 1 );
              }
            } // norm
          } // par

          if( _owner_->getPropagator().isThrowAsimovToyParameters() ){
            bool draw{false};

            for( auto& par : parList_ ){
              double val{par.getThrowValue()};
              val == val ? draw = true : val = par.getPriorValue();
              if( isNorm_ ) val = FitParameterSet::toNormalizedParValue(val, par);
              toyParametersLine->SetBinContent(1+par.getParameterIndex(), val);
            }

            if( !draw ) toyParametersLine = nullptr;
          }

          auto yBounds = GenericToolbox::getYBounds({preFitErrorHist.get(), postFitErrorHist.get(), toyParametersLine.get()});

          for( const auto& par : parList_ ){
            TBox b(preFitErrorHist->GetBinLowEdge(1+par.getParameterIndex()), yBounds.first,
                   preFitErrorHist->GetBinLowEdge(1+par.getParameterIndex()+1), yBounds.second);
            b.SetFillStyle(3001);

            if( par.isFree() ){
              b.SetFillColor(kGreen-10);
              freeParBoxes.emplace_back(b);
            }
            else if( par.isFixed() or not par.isEnabled() ){
              b.SetFillColor(kGray);
              fixedParBoxes.emplace_back(b);
            }
          }

          if(parSet_.getPriorCovarianceMatrix() != nullptr ){
            gStyle->GetCanvasPreferGL() ? preFitErrorHist->SetFillColorAlpha(kRed-9, 0.7) : preFitErrorHist->SetFillColor(kRed-9);
          }

          preFitErrorHist->SetMarkerStyle(kFullDotLarge);
          preFitErrorHist->SetMarkerColor(kRed-3);
          preFitErrorHist->SetLineColor(kRed-3); // for legend

          if( not isNorm_ ){
            preFitErrorHist->GetYaxis()->SetTitle("Parameter values (a.u.)");
          }
          else{
            preFitErrorHist->GetYaxis()->SetTitle("Parameter values (normalized to the prior)");
          }
          preFitErrorHist->GetXaxis()->SetLabelSize(0.03);
          preFitErrorHist->GetXaxis()->LabelsOption("v");

          preFitErrorHist->SetTitle(Form("Pre-fit Errors of %s", parSet_.getName().c_str()));
          preFitErrorHist->SetMarkerSize(0);
          preFitErrorHist->GetYaxis()->SetRangeUser(yBounds.first, yBounds.second);
          GenericToolbox::writeInTFile(saveDir_, preFitErrorHist.get());

          postFitErrorHist->SetLineColor(9);
          postFitErrorHist->SetLineWidth(2);
          postFitErrorHist->SetMarkerColor(9);
          postFitErrorHist->SetMarkerStyle(kFullDotLarge);
          postFitErrorHist->SetTitle(Form("Post-fit Errors of %s", parSet_.getName().c_str()));
          GenericToolbox::writeInTFile(saveDir_, postFitErrorHist.get());

          auto errorsCanvas = std::make_unique<TCanvas>(
              Form("Fit Constraints for %s", parSet_.getName().c_str()),
              Form("Fit Constraints for %s", parSet_.getName().c_str()),
              800, 600);
          errorsCanvas->cd();

          preFitErrorHist->Draw("E2");

          for( auto& box : freeParBoxes ) box.Draw();
          for( auto& box : fixedParBoxes ) box.Draw();
          preFitErrorHist->Draw("E2 SAME");

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

          if( toyParametersLine != nullptr ){
            legend->SetY1(legend->GetY1() - (legend->GetY2() - legend->GetY1())/2.);
            legend->AddEntry(toyParametersLine.get(),"Toy throws from asimov data set","l");
            toyParametersLine->SetLineColor(kGray+2);
            toyParametersLine->Draw("SAME");
          }

          errorsCanvas->Update(); // otherwise does not display...
          postFitErrorHist->Draw("E1 X0 SAME");

          legend->Draw();

          gPad->SetGridx();
          gPad->SetGridy();
          gPad->SetBottomMargin(float(0.1*(1. + double(longestTitleSize)/15.)));

          if( not isNorm_ ){ preFitErrorHist->SetTitle(Form("Pre-fit/Post-fit comparison for %s", parSet_.getName().c_str())); }
          else             { preFitErrorHist->SetTitle(Form("Pre-fit/Post-fit comparison for %s (normalized)", parSet_.getName().c_str())); }
          GenericToolbox::writeInTFile(saveDir_, errorsCanvas.get(), "fitConstraints");

        }; // makePrePostFitCompPlot

        makePrePostFitCompPlot(GenericToolbox::mkdirTFile(saveSubdir_, "values"), false);
        makePrePostFitCompPlot(GenericToolbox::mkdirTFile(saveSubdir_, "valuesNorm"), true);

      }; // savePostFitObjFct

  LogInfo << "Extracting post-fit errors..." << std::endl;
  for( const auto& parSet : _owner_->getPropagator().getParameterSetsList() ){
    if( not parSet.isEnabled() ){ continue; }

    LogWarning << "Extracting post-fit errors of parameter set: " << parSet.getName() << std::endl;
    auto* parSetDir = GenericToolbox::mkdirTFile(errorDir, parSet.getName());

    auto* parList = &parSet.getEffectiveParameterList();
    // dimension should be the right one -> parList includes the fixed one
    auto covMatrix = std::make_unique<TMatrixD>(int(parList->size()), int(parList->size()));
    for( auto& iPar : *parList ){
      int iMinimizerIndex = GenericToolbox::findElementIndex((FitParameter*) &iPar, _minimizerFitParameterPtr_);
      if( iMinimizerIndex == -1 ) continue;
      for( auto& jPar : *parList ){
        int jMinimizerIndex = GenericToolbox::findElementIndex((FitParameter*) &jPar, _minimizerFitParameterPtr_);
        if( jMinimizerIndex == -1 ) continue;
        (*covMatrix)[iPar.getParameterIndex()][jPar.getParameterIndex()] = postfitCovarianceMatrix[iMinimizerIndex][jMinimizerIndex];
      }
    }

    TDirectory* saveDir;
    if( parSet.isUseEigenDecompInFit() ){
      saveDir = GenericToolbox::mkdirTFile(parSetDir, "eigen");
      savePostFitObjFct(parSet, *parList, covMatrix.get(), saveDir);

      // need to restore the non-fitted values before the base swap
      for( auto& eigenPar : *parList ){
        if( eigenPar.isEnabled() and not eigenPar.isFixed() ) continue;
        (*covMatrix)[eigenPar.getParameterIndex()][eigenPar.getParameterIndex()] = eigenPar.getStdDevValue() * eigenPar.getStdDevValue();
      }

      // The actual base conversion is here
      auto originalStrippedCovMatrix = std::make_unique<TMatrixD>(covMatrix->GetNrows(), covMatrix->GetNcols());
      (*originalStrippedCovMatrix) =  (*parSet.getEigenVectors());
      (*originalStrippedCovMatrix) *= (*covMatrix);
      (*originalStrippedCovMatrix) *= (*parSet.getInvertedEigenVectors());

      // force real parameters
      parList = &parSet.getParameterList();

      // restore the original size of the matrix
      covMatrix = std::make_unique<TMatrixD>(int(parList->size()), int(parList->size()));
      int iStripped{-1};
      for( auto& iPar : *parList ){
        if( iPar.isFixed() or not iPar.isEnabled() ) continue;
        iStripped++;
        int jStripped{-1};
        for( auto& jPar : *parList ){
          if( jPar.isFixed() or not jPar.isEnabled() ) continue;
          jStripped++;
          (*covMatrix)[iPar.getParameterIndex()][jPar.getParameterIndex()] = (*originalStrippedCovMatrix)[iStripped][jStripped];
        }
      }
    }

    savePostFitObjFct(parSet, *parList, covMatrix.get(), parSetDir);

  } // parSet
}


