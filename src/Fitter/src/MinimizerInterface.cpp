//
// Created by Adrien BLANCHET on 16/12/2021.
//

#include "LikelihoodInterface.h"
#include "MinimizerInterface.h"
#include "FitterEngine.h"
#include "GenericToolbox.Json.h"
#include "GundamGlobals.h"
#include "GundamUtils.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"

#include "Math/Factory.h"
#include "Math/Minimizer.h"
#include "Math/Functor.h"
#include "TLegend.h"


LoggerInit([]{
  Logger::setUserHeaderStr("[MinimizerInterface]");
});


void MinimizerInterface::readConfigImpl(){
  MinimizerBase::readConfigImpl();
  LogInfo << "Reading minimizer config..." << std::endl;

  _minimizerType_ = GenericToolbox::Json::fetchValue(_config_, "minimizer", _minimizerType_);
  _minimizerAlgo_ = GenericToolbox::Json::fetchValue(_config_, "algorithm", _minimizerAlgo_);

  _strategy_ = GenericToolbox::Json::fetchValue(_config_, "strategy", _strategy_);
  _printLevel_ = GenericToolbox::Json::fetchValue(_config_, "print_level", _printLevel_);
  _tolerance_ = GenericToolbox::Json::fetchValue(_config_, "tolerance", _tolerance_);
  _maxIterations_ = GenericToolbox::Json::fetchValue(_config_, {{"maxIterations"}, {"max_iter"}}, _maxIterations_ );
  _maxFcnCalls_ = GenericToolbox::Json::fetchValue(_config_, {{"maxFcnCalls"}, {"max_fcn"}}, _maxFcnCalls_ );

  _enableSimplexBeforeMinimize_ = GenericToolbox::Json::fetchValue(_config_, "enableSimplexBeforeMinimize", _enableSimplexBeforeMinimize_);
  _simplexMaxFcnCalls_ = GenericToolbox::Json::fetchValue(_config_, "simplexMaxFcnCalls", _simplexMaxFcnCalls_);
  _simplexToleranceLoose_ = GenericToolbox::Json::fetchValue(_config_, "simplexToleranceLoose", _simplexToleranceLoose_);
  _simplexStrategy_ = GenericToolbox::Json::fetchValue(_config_, "simplexStrategy", _simplexStrategy_);

  _errorAlgo_ = GenericToolbox::Json::fetchValue(_config_, {{"errorsAlgo"}, {"errors"}}, "Hesse");
  _restoreStepSizeBeforeHesse_ = GenericToolbox::Json::fetchValue(_config_, "restoreStepSizeBeforeHesse", _restoreStepSizeBeforeHesse_);

  _generatedPostFitParBreakdown_ = GenericToolbox::Json::fetchValue(_config_, "generatedPostFitParBreakdown", _generatedPostFitParBreakdown_);
  _generatedPostFitEigenBreakdown_ = GenericToolbox::Json::fetchValue(_config_, "generatedPostFitEigenBreakdown", _generatedPostFitEigenBreakdown_);

  _stepSizeScaling_ = GenericToolbox::Json::fetchValue(_config_, "stepSizeScaling", _stepSizeScaling_);

  if( GenericToolbox::Json::doKeyExist( _config_, "monitorGradientDescent" ) ){
    LogInfo << "Will monitor the gradient descent." << std::endl;
    getLikelihood().setMonitorGradientDescent(GenericToolbox::Json::fetchValue<bool>( _config_, "monitorGradientDescent" ));
  }

}
void MinimizerInterface::initializeImpl(){
  MinimizerBase::initializeImpl();
  LogInfo << "Initializing the minimizer..." << std::endl;

  _minimizer_ = std::unique_ptr<ROOT::Math::Minimizer>(
      ROOT::Math::Factory::CreateMinimizer(_minimizerType_, _minimizerAlgo_)
  );
  LogThrowIf(_minimizer_ == nullptr, "Could not create minimizer: " << _minimizerType_ << "/" << _minimizerAlgo_)
  if( _minimizerAlgo_.empty() ){
    _minimizerAlgo_ = _minimizer_->Options().MinimizerAlgorithm();
    LogWarning << "Using default minimizer algo: " << _minimizerAlgo_ << std::endl;
  }

  // Configure the likelihood with the local settings.  These are used by the
  // likelihood to print informational messages, but do not affect how the
  // likelihood code runs.
  getLikelihood().setMinimizerInfo(_minimizerType_,_minimizerAlgo_);
  getLikelihood().setTargetEDM(0.001*_tolerance_*2);

  _minimizer_->SetFunction(*(getLikelihood().evalFitFunctor()));
  _minimizer_->SetStrategy(_strategy_);
  _minimizer_->SetPrintLevel(_printLevel_);
  _minimizer_->SetTolerance(_tolerance_);
  _minimizer_->SetMaxIterations(_maxIterations_);
  _minimizer_->SetMaxFunctionCalls(_maxFcnCalls_);

  for( std::size_t iFitPar = 0 ; iFitPar < getMinimizerFitParameterPtr().size() ; iFitPar++ ){
    auto& fitPar = *(getMinimizerFitParameterPtr()[iFitPar]);

    if( not getLikelihood().getUseNormalizedFitSpace() ){
      _minimizer_->SetVariable(iFitPar, fitPar.getFullTitle(),fitPar.getParameterValue(),fitPar.getStepSize()*_stepSizeScaling_);
      if( not std::isnan( fitPar.getMinValue() ) ){ _minimizer_->SetVariableLowerLimit(iFitPar, fitPar.getMinValue()); }
      if( not std::isnan( fitPar.getMaxValue() ) ){ _minimizer_->SetVariableUpperLimit(iFitPar, fitPar.getMaxValue()); }
      // Changing the boundaries, change the value/step size?
      _minimizer_->SetVariableValue(iFitPar, fitPar.getParameterValue());
      _minimizer_->SetVariableStepSize(iFitPar, fitPar.getStepSize()*_stepSizeScaling_);
    }
    else{
      _minimizer_->SetVariable(iFitPar, fitPar.getFullTitle(),
                               ParameterSet::toNormalizedParValue(fitPar.getParameterValue(), fitPar),
                               ParameterSet::toNormalizedParRange(fitPar.getStepSize() * _stepSizeScaling_, fitPar)
      );
      if( not std::isnan( fitPar.getMinValue() ) ){ _minimizer_->SetVariableLowerLimit(iFitPar, ParameterSet::toNormalizedParValue(fitPar.getMinValue(), fitPar)); }
      if( not std::isnan( fitPar.getMaxValue() ) ){ _minimizer_->SetVariableUpperLimit(iFitPar, ParameterSet::toNormalizedParValue(fitPar.getMaxValue(), fitPar)); }
      // Changing the boundaries, change the value/step size?
      _minimizer_->SetVariableValue(iFitPar, ParameterSet::toNormalizedParValue(fitPar.getParameterValue(), fitPar));
      _minimizer_->SetVariableStepSize(iFitPar, ParameterSet::toNormalizedParRange(fitPar.getStepSize() * _stepSizeScaling_, fitPar));
    }
  }

}

// overridden getters
double MinimizerInterface::getTargetEdm() const{
  // Migrad: The default tolerance is 0.1, and the minimization will stop
  // when the estimated vertical distance to the minimum (EDM) is less
  // than 0.001*[tolerance]*UP (see SET ERR).
  // UP:
  // Minuit defines parameter errors as the change in parameter value required
  // to change the function value by UP. Normally, for chisquared fits
  // UP=1, and for negative log likelihood, UP=0.5
  return 0.001 * _tolerance_ * 1;
}

// core overrides
void MinimizerInterface::minimize(){
  LogThrowIf(not isInitialized(), "not initialized");

  printMinimizerFitParameters();

  getPropagator().updateLlhCache();

  LogWarning << std::endl << GenericToolbox::addUpDownBars("Calling minimize...") << std::endl;
  LogInfo << "Number of defined parameters: " << _minimizer_->NDim() << std::endl
          << "Number of fit parameters   : " << _minimizer_->NFree() << std::endl
          << "Number of free parameters : " << getLikelihood().getNbFreePars() << std::endl
          << "Number of fixed parameters  : " << _minimizer_->NDim() - _minimizer_->NFree() << std::endl
          << "Number of fit bins : " << getLikelihood().getNbFitBins() << std::endl
          << "Chi2 # DoF : " << getLikelihood().getNbFitBins() - getLikelihood().getNbFreePars()
          << std::endl;

  int nbFitCallOffset = getLikelihood().getNbFitCalls();
  LogInfo << "Fit call offset: " << nbFitCallOffset << std::endl;

  if( _enableSimplexBeforeMinimize_ ){
    LogWarning << "Running simplex algo before the minimizer" << std::endl;
    LogThrowIf(_minimizerType_ != "Minuit2", "Can't launch simplex with " << _minimizerType_);

    std::string originalAlgo = _minimizer_->Options().MinimizerAlgorithm();

    _minimizer_->Options().SetMinimizerAlgorithm("Simplex");
    _minimizer_->SetMaxFunctionCalls(_simplexMaxFcnCalls_);
    _minimizer_->SetTolerance( _tolerance_ * _simplexToleranceLoose_ );
    _minimizer_->SetStrategy(0);

    getLikelihood().setStateTitleMonitor("Running Simplex...");

    // SIMPLEX
    getLikelihood().enableFitMonitor();
    _fitHasConverged_ = _minimizer_->Minimize();
    getLikelihood().disableFitMonitor();

    // Make sure we are on the right spot
    updateCacheToBestfitPoint();

    // export bf point with SIMPLEX
    LogInfo << "Writing " << _minimizerType_ << "/Simplex best fit parameters..." << std::endl;
    GenericToolbox::writeInTFile(
        GenericToolbox::mkdirTFile( owner().getSaveDir(), GenericToolbox::joinPath("postFit", _minimizerAlgo_) ),
        TNamed("parameterStateAfterSimplex", GenericToolbox::Json::toReadableString( getPropagator().getParametersManager().exportParameterInjectorConfig() ).c_str() )
    );

    // Back to original
    _minimizer_->Options().SetMinimizerAlgorithm(originalAlgo.c_str());
    _minimizer_->SetMaxFunctionCalls(_maxFcnCalls_);
    _minimizer_->SetTolerance(_tolerance_);
    _minimizer_->SetStrategy(_strategy_);

    LogInfo << getConvergenceMonitor().generateMonitorString(); // lasting printout
    LogWarning << "Simplex ended after " << getLikelihood().getNbFitCalls() - nbFitCallOffset << " calls." << std::endl;
  }

  getLikelihood().setStateTitleMonitor("Running " + _minimizer_->Options().MinimizerAlgorithm() + "...");

  getLikelihood().enableFitMonitor();
  _fitHasConverged_ = _minimizer_->Minimize();
  getLikelihood().disableFitMonitor();

  int nbMinimizeCalls = getLikelihood().getNbFitCalls() - nbFitCallOffset;

  LogInfo << getConvergenceMonitor().generateMonitorString(); // lasting printout
  LogInfo << "Minimization ended after " << nbMinimizeCalls << " calls." << std::endl;
  if(_minimizerType_ == "Minuit" or _minimizerType_ == "Minuit2") LogWarning << "Status code: " << GundamUtils::minuitStatusCodeStr.at(_minimizer_->Status()) << std::endl;
  else LogWarning << "Status code: " << _minimizer_->Status() << std::endl;
  if(_minimizerType_ == "Minuit" or _minimizerType_ == "Minuit2") LogWarning << "Covariance matrix status code: " << GundamUtils::covMatrixStatusCodeStr.at(_minimizer_->CovMatrixStatus()) << std::endl;
  else LogWarning << "Covariance matrix status code: " << _minimizer_->CovMatrixStatus() << std::endl;

  // Make sure we are on the right spot
  updateCacheToBestfitPoint();

  // export bf point
  LogInfo << "Writing " << _minimizerType_ << "/" << _minimizerAlgo_ << " best fit parameters..." << std::endl;
  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile( owner().getSaveDir(), GenericToolbox::joinPath("postFit", _minimizerAlgo_) ),
      TNamed("parameterStateAfterMinimize", GenericToolbox::Json::toReadableString( getPropagator().getParametersManager().exportParameterInjectorConfig() ).c_str() )
  );

  getLikelihood().saveChi2History();
  if( getLikelihood().isMonitorGradientDescent() ){ getLikelihood().saveGradientSteps(); }

  if( _fitHasConverged_ ){ LogInfo << "Minimization has converged!" << std::endl; }
  else{ LogError << "Minimization did not converged." << std::endl; }


  LogInfo << "Writing convergence stats..." << std::endl;
  int toyIndex = getPropagator().getIThrow();
  int nIterations = int(_minimizer_->NIterations());
  int nFitPars = int(_minimizer_->NFree());
  int nDof = getLikelihood().getNbFitBins() - getLikelihood().getNbFreePars();
  double edmBestFit = _minimizer_->Edm();
  double fitStatus = _minimizer_->Status();
  double covStatus = _minimizer_->CovMatrixStatus();
  double chi2MinFitter = _minimizer_->MinValue();
  int nbFreePars = getLikelihood().getNbFreePars();
  int nbFitCalls = getLikelihood().getNbFitCalls();
  int nbFitBins = getLikelihood().getNbFitBins();

  auto bestFitStats = std::make_unique<TTree>("bestFitStats", "bestFitStats");
  bestFitStats->SetDirectory(nullptr);
  bestFitStats->Branch("fitConverged", &_fitHasConverged_);
  bestFitStats->Branch("fitStatusCode", &fitStatus);
  bestFitStats->Branch("covStatusCode", &covStatus);
  bestFitStats->Branch("edmBestFit", &edmBestFit);
  bestFitStats->Branch("nIterations", &nIterations);
  bestFitStats->Branch("chi2MinFitter", &chi2MinFitter);
  bestFitStats->Branch("toyIndex", &toyIndex);
  bestFitStats->Branch("nCallsAtBestFit", &nbFitCalls);
  bestFitStats->Branch("nFitBins", &nbFitBins);
  bestFitStats->Branch("nFreePars", &nbFreePars);
  bestFitStats->Branch("nFitPars", &nFitPars);
  bestFitStats->Branch("nDof", &nDof);
  bestFitStats->Branch("chi2BestFit",      (double*) getPropagator().getLlhBufferPtr() ); // need non const ptr...
  bestFitStats->Branch("chi2StatBestFit",  (double*) getPropagator().getLlhStatBufferPtr() );
  bestFitStats->Branch("chi2PullsBestFit", (double*) getPropagator().getLlhPenaltyBufferPtr() );

  std::vector<GenericToolbox::RawDataArray> samplesArrList(getPropagator().getSampleSet().getSampleList().size());
  int iSample{-1};
  for( auto& sample : getPropagator().getSampleSet().getSampleList() ){
    if( not sample.isEnabled() ) continue;

    std::vector<std::string> leavesDict;
    iSample++;

    leavesDict.emplace_back("llhSample/D");
    samplesArrList[iSample].writeRawData(getPropagator().getSampleSet().getJointProbabilityFct()->eval(sample));

    int nBins = int(sample.getBinning().getBinList().size());
    for( int iBin = 1 ; iBin <= nBins ; iBin++ ){
      leavesDict.emplace_back("llhSample_bin" + std::to_string(iBin) + "/D");
      samplesArrList[iSample].writeRawData(getPropagator().getSampleSet().getJointProbabilityFct()->eval(sample, iBin));
    }

    samplesArrList[iSample].lockArraySize();
    bestFitStats->Branch(
        GenericToolbox::generateCleanBranchName(sample.getName()).c_str(),
        &samplesArrList[iSample].getRawDataArray()[0],
        GenericToolbox::joinVectorString(leavesDict, ":").c_str()
    );
  }

  std::vector<GenericToolbox::RawDataArray> parameterSetArrList(getPropagator().getParametersManager().getParameterSetsList().size());
  int iParSet{-1};
  for( auto& parSet : getPropagator().getParametersManager().getParameterSetsList() ){
    if( not parSet.isEnabled() ) continue;

    std::vector<std::string> leavesDict;
    iParSet++;

    leavesDict.emplace_back("llhPenalty/D");
    parameterSetArrList[iParSet].writeRawData(parSet.getPenaltyChi2());

    for( auto& par : parSet.getParameterList() ){
      leavesDict.emplace_back(GenericToolbox::replaceSubstringInString(par.getTitle(), " ", "_") + "/D");
      parameterSetArrList[iParSet].writeRawData(par.getParameterValue());
    }

    bestFitStats->Branch(
        GenericToolbox::generateCleanBranchName(parSet.getName()).c_str(),
        &parameterSetArrList[iParSet].getRawDataArray()[0],
        GenericToolbox::joinVectorString(leavesDict, ":").c_str()
    );
  }

  bestFitStats->Fill();
  GenericToolbox::mkdirTFile(owner().getSaveDir(), "postFit")->WriteObject(bestFitStats.get(), bestFitStats->GetName());

  LogInfo << "Writing " << _minimizerType_ << "/" << _minimizerAlgo_ << " post-fit errors" << std::endl;
  this->writePostFitData(GenericToolbox::mkdirTFile(owner().getSaveDir(), GenericToolbox::joinPath("postFit", _minimizerAlgo_)));
  GenericToolbox::triggerTFileWrite(GenericToolbox::mkdirTFile(owner().getSaveDir(), GenericToolbox::joinPath("postFit", _minimizerAlgo_)));
}
void MinimizerInterface::calcErrors(){
  LogThrowIf(not isInitialized(), "not initialized");

  int nbFitCallOffset = getLikelihood().getNbFitCalls();

  LogWarning << std::endl << GenericToolbox::addUpDownBars("Calling HESSE...") << std::endl;
  LogInfo << "Number of defined parameters: " << _minimizer_->NDim() << std::endl
          << "Number of fit parameters   : " << _minimizer_->NFree() << std::endl
          << "Number of free parameters : " << getLikelihood().getNbFreePars() << std::endl
          << "Number of fixed parameters  : " << _minimizer_->NDim() - _minimizer_->NFree() << std::endl
          << "Number of fit bins : " << getLikelihood().getNbFitBins() << std::endl
          << "Chi2 # DoF : " << getLikelihood().getNbFitBins() - getLikelihood().getNbFreePars()  << std::endl
          << "Fit call offset: " << nbFitCallOffset << std::endl;

  if     ( _errorAlgo_ == "Minos" ){
    LogWarning << std::endl << GenericToolbox::addUpDownBars("Calling MINOS...") << std::endl;

    double errLow, errHigh;
    _minimizer_->SetPrintLevel(0);

    for( int iFitPar = 0 ; iFitPar < _minimizer_->NDim() ; iFitPar++ ){
      LogInfo << "Evaluating: " << _minimizer_->VariableName(iFitPar) << "..." << std::endl;

      getLikelihood().enableFitMonitor();
      bool isOk = _minimizer_->GetMinosError(iFitPar, errLow, errHigh);
      getLikelihood().disableFitMonitor();

#if ROOT_VERSION_CODE >= ROOT_VERSION(6,23,02)
      LogWarning << GundamUtils::minosStatusCodeStr.at(_minimizer_->MinosStatus()) << std::endl;
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
      getMinimizerFitParameterPtr()[iFitPar]->setParameterValue(_minimizer_->X()[iFitPar]);
    }
  } // Minos
  else if( _errorAlgo_ == "Hesse" ){

    if( _restoreStepSizeBeforeHesse_ ){
      LogWarning << "Restoring step size before HESSE..." << std::endl;
      for( int iFitPar = 0 ; iFitPar < _minimizer_->NDim() ; iFitPar++ ){
        auto& par = *getMinimizerFitParameterPtr()[iFitPar];
        if(not getLikelihood().getUseNormalizedFitSpace()){ _minimizer_->SetVariableStepSize(iFitPar, par.getStepSize()*_stepSizeScaling_); }
        else{ _minimizer_->SetVariableStepSize(iFitPar, ParameterSet::toNormalizedParRange(par.getStepSize() * _stepSizeScaling_, par)); } // should be 1
      }
    }

    // Make sure we are on the right spot
    updateCacheToBestfitPoint();

    getLikelihood().setStateTitleMonitor("Running HESSE...");

    getLikelihood().enableFitMonitor();
    _fitHasConverged_ = _minimizer_->Hesse();
    getLikelihood().disableFitMonitor();

    LogInfo << "Hesse ended after " << getLikelihood().getNbFitCalls() - nbFitCallOffset << " calls." << std::endl;
    LogWarning << "HESSE status code: " << GundamUtils::hesseStatusCodeStr.at(_minimizer_->Status()) << std::endl;
    LogWarning << "Covariance matrix status code: " << GundamUtils::covMatrixStatusCodeStr.at(_minimizer_->CovMatrixStatus()) << std::endl;

    // Make sure we are on the right spot
    updateCacheToBestfitPoint();

    if(not _fitHasConverged_){
      LogError  << "Hesse did not converge." << std::endl;
      LogError << getConvergenceMonitor().generateMonitorString(); // lasting printout
    }
    else{
      LogInfo << "Hesse converged." << std::endl;
      LogInfo << getConvergenceMonitor().generateMonitorString(); // lasting printout
    }

    int covStatus = _minimizer_->CovMatrixStatus();

    auto hesseStats = std::make_unique<TTree>("hesseStats", "hesseStats");
    hesseStats->SetDirectory(nullptr);
    hesseStats->Branch("hesseSuccess", &_fitHasConverged_);
    hesseStats->Branch("covStatusCode", &covStatus);

    hesseStats->Fill();
    GenericToolbox::mkdirTFile(owner().getSaveDir(), "postFit")->WriteObject(hesseStats.get(), hesseStats->GetName());

    LogInfo << "Writing HESSE post-fit errors" << std::endl;
    this->writePostFitData(GenericToolbox::mkdirTFile(owner().getSaveDir(), "postFit/Hesse"));
    GenericToolbox::triggerTFileWrite(GenericToolbox::mkdirTFile(owner().getSaveDir(), "postFit/Hesse"));
  }
  else{
    LogError << GET_VAR_NAME_VALUE(_errorAlgo_) << " not implemented." << std::endl;
  }
}
void MinimizerInterface::scanParameters(TDirectory* saveDir_){
  LogThrowIf(not isInitialized());
  LogInfo << "Performing scans of fit parameters..." << std::endl;
  for( int iPar = 0 ; iPar < getMinimizer()->NDim() ; iPar++ ){
    if( getMinimizer()->IsFixedVariable(iPar) ){
      LogWarning << getMinimizer()->VariableName(iPar)
                 << " is fixed. Skipping..." << std::endl;
      continue;
    }
    this->getPropagator().getParScanner().scanFitParameter(*getMinimizerFitParameterPtr()[iPar], saveDir_);
  } // iPar
  for( auto& parSet : this->getPropagator().getParametersManager().getParameterSetsList() ){
    if( not parSet.isEnabled() ) continue;
    if( parSet.isUseEigenDecompInFit() ){
      LogWarning << parSet.getName() << " is using eigen decomposition. Scanning original parameters..." << std::endl;
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ) continue;
        this->getPropagator().getParScanner().scanFitParameter(par, saveDir_);
      }
    }
  }
}

// misc
void MinimizerInterface::saveMinimizerSettings(TDirectory* saveDir_) const {
  LogInfo << "Saving minimizer settings..." << std::endl;

  GenericToolbox::writeInTFile( saveDir_, TNamed("minimizerType", _minimizerType_.c_str()) );
  GenericToolbox::writeInTFile( saveDir_, TNamed("minimizerAlgo", _minimizerAlgo_.c_str()) );
  GenericToolbox::writeInTFile( saveDir_, TNamed("strategy", std::to_string(_strategy_).c_str()) );
  GenericToolbox::writeInTFile( saveDir_, TNamed("printLevel", std::to_string(_printLevel_).c_str()) );
  GenericToolbox::writeInTFile( saveDir_, TNamed("targetEDM", std::to_string(this->getTargetEdm()).c_str()) );
  GenericToolbox::writeInTFile( saveDir_, TNamed("maxIterations", std::to_string(_maxIterations_).c_str()) );
  GenericToolbox::writeInTFile( saveDir_, TNamed("maxFcnCalls", std::to_string(_maxFcnCalls_).c_str()) );
  GenericToolbox::writeInTFile( saveDir_, TNamed("tolerance", std::to_string(_tolerance_).c_str()) );
  GenericToolbox::writeInTFile( saveDir_, TNamed("stepSizeScaling", std::to_string(_stepSizeScaling_).c_str()) );
  GenericToolbox::writeInTFile( saveDir_, TNamed("useNormalizedFitSpace", std::to_string(getLikelihood().getUseNormalizedFitSpace()).c_str()) );

  if( _enableSimplexBeforeMinimize_ ){
    GenericToolbox::writeInTFile( saveDir_, TNamed("enableSimplexBeforeMinimize", std::to_string(_enableSimplexBeforeMinimize_).c_str()) );
    GenericToolbox::writeInTFile( saveDir_, TNamed("simplexMaxFcnCalls", std::to_string(_simplexMaxFcnCalls_).c_str()) );
    GenericToolbox::writeInTFile( saveDir_, TNamed("simplexToleranceLoose", std::to_string(_simplexToleranceLoose_).c_str()) );
    GenericToolbox::writeInTFile( saveDir_, TNamed("simplexStrategy", std::to_string(_simplexStrategy_).c_str()) );
  }

  if( this->isEnablePostFitErrorEval() ){
    GenericToolbox::writeInTFile( saveDir_, TNamed("enablePostFitErrorFit", std::to_string(this->isEnablePostFitErrorEval()).c_str()) );
    GenericToolbox::writeInTFile( saveDir_, TNamed("errorAlgo", _errorAlgo_.c_str()) );
  }
}

// protected
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
      if( not GundamGlobals::isLightOutputMode() ) {
        GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(outDir_, "eigenDecomposition"), eigenVectors.get(), "eigenVectors");
      }

      auto eigenValues = std::unique_ptr<TH1D>( GenericToolbox::convertTVectorDtoTH1D(&decompCovMatrix.GetEigenValues()) );
      applyBinLabels(eigenValues.get());
      if( not GundamGlobals::isLightOutputMode() ) {
        GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(outDir_, "eigenDecomposition"), eigenValues.get(), "eigenValues");
      }

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
      if( not GundamGlobals::isLightOutputMode() ){
        GenericToolbox::writeInTFile(outDir_, postfitHessianTH2D, "postfitHessian");
      }

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

        // normalize to the maximum hist
        double minYval{std::nan("")};
        for (int iEigen = decompCovMatrix.GetEigenValues().GetNrows() - 1; iEigen >= 0; iEigen--) {
          for (int iPar = 0 ; iPar < eigenBreakdownAccum[iEigen].GetNbinsX() ; iPar++ ) {
            eigenBreakdownAccum[iEigen].SetBinContent(
                iPar + 1,
                eigenBreakdownAccum[iEigen].GetBinContent(iPar + 1)
                /eigenBreakdownAccum[0].GetBinContent(iPar + 1)
            );
            if( std::isnan(minYval) ){ minYval = eigenBreakdownAccum[iEigen].GetBinContent(iPar + 1); }
            else{
              minYval = std::min(minYval, eigenBreakdownAccum[iEigen].GetBinContent(iPar + 1));
            }
          }
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
            eigenBreakdownAccum[iEigen].GetYaxis()->SetRangeUser(minYval, eigenBreakdownAccum[iEigen].GetMaximum()*1.2);
            eigenBreakdownAccum[iEigen].GetYaxis()->SetTitle("Hessian eigen composition of postfit errors");
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

        if( not GundamGlobals::isLightOutputMode() ) {
          GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(outDir_, "eigenDecomposition"), &accumPlot, "eigenBreakdown");
        }
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

        if( not GundamGlobals::isLightOutputMode() ) {
          GenericToolbox::writeInTFile(
              GenericToolbox::mkdirTFile(outDir_, "eigenDecomposition"),
              &accumPlot, "parBreakdown"
          );
        }
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
      for( const auto& parSet : getPropagator().getParametersManager().getParameterSetsList() ){ if( parSet.isEnabled() ) nGlobalPars += int(parSet.getNbParameters()); }

      // Reconstruct the global passage matrix
      std::vector<std::string> parameterLabels(nGlobalPars);
      auto globalPassageMatrix = std::make_unique<TMatrixD>(nGlobalPars, nGlobalPars);
      for(int i = 0 ; i < nGlobalPars; i++ ){ (*globalPassageMatrix)[i][i] = 1; }
      int blocOffset{0};
      for( const auto& parSet : getPropagator().getParametersManager().getParameterSetsList() ){
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
      for( const auto& iParSet : getPropagator().getParametersManager().getParameterSetsList() ){
        if( not iParSet.isEnabled() ) continue;

        auto* iParList = &iParSet.getEffectiveParameterList();
        for( auto& iPar : *iParList ){
          int iMinimizerIndex = GenericToolbox::findElementIndex((Parameter*) &iPar, getMinimizerFitParameterPtr());

          int jOffset{0};
          for( const auto& jParSet : getPropagator().getParametersManager().getParameterSetsList() ){
            if( not jParSet.isEnabled() ) continue;

            auto* jParList = &jParSet.getEffectiveParameterList();
            for( auto& jPar : *jParList ){
              int jMinimizerIndex = GenericToolbox::findElementIndex((Parameter*) &jPar,
                                                                     getMinimizerFitParameterPtr());

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

  if( getLikelihood().getUseNormalizedFitSpace() ){
    LogInfo << "Writing normalized decomposition of the output matrix..." << std::endl;
    if( not GundamGlobals::isLightOutputMode() ) {
      decomposeCovarianceMatrixFct(GenericToolbox::mkdirTFile(matricesDir, "normalizedFitSpace"));
    }

    // Rescale the post-fit values:
    for(int iRow = 0 ; iRow < postfitCovarianceMatrix.GetNrows() ; iRow++ ){
      for(int iCol = 0 ; iCol < postfitCovarianceMatrix.GetNcols() ; iCol++ ){
        postfitCovarianceMatrix[iRow][iCol] *= (getMinimizerFitParameterPtr()[iRow]->getStdDevValue()) * (getMinimizerFitParameterPtr()[iCol]->getStdDevValue());
      }
    }

  }

  LogInfo << "Writing decomposition of the output matrix..." << std::endl;
  decomposeCovarianceMatrixFct(matricesDir);

  LogInfo << "Fitter covariance matrix is " << postfitCovarianceMatrix.GetNrows() << "x" << postfitCovarianceMatrix.GetNcols() << std::endl;
  auto* errorDir = GenericToolbox::mkdirTFile(saveDir_, "errors");

  auto savePostFitObjFct =
      [&](const ParameterSet& parSet_, const std::vector<Parameter>& parList_, TMatrixD* covMatrix_, TDirectory* saveSubdir_){
        GenericToolbox::mkdirTFile(saveSubdir_, "matrices")->cd(); // prevent ROOT to delete other hists with the same name...

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
                  ParameterSet::toNormalizedParValue(par.getParameterValue(), par)
              );
              preFitErrorHist->SetBinContent( 1 + par.getParameterIndex(), 0 );

              postFitErrorHist->SetBinError(
                  1 + par.getParameterIndex(),
                  ParameterSet::toNormalizedParRange(
                      TMath::Sqrt((*covMatrix_)[par.getParameterIndex()][par.getParameterIndex()]), par
                  )
              );
              if( par.isEnabled() and not par.isFixed() and not par.isFree() ){
                preFitErrorHist->SetBinError( 1 + par.getParameterIndex(), 1 );
              }
            } // norm
          } // par

          if( getPropagator().isThrowAsimovToyParameters() ){
            bool draw{false};

            for( auto& par : parList_ ){
              double val{par.getThrowValue()};
              val == val ? draw = true : val = par.getPriorValue();
              if( isNorm_ ) val = ParameterSet::toNormalizedParValue(val, par);
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
          gPad->SetBottomMargin(float(0.1*(1. + double(longestTitleSize)/12.)));

          if( not isNorm_ ){ preFitErrorHist->SetTitle(Form("Pre-fit/Post-fit comparison for %s", parSet_.getName().c_str())); }
          else             { preFitErrorHist->SetTitle(Form("Pre-fit/Post-fit comparison for %s (normalized)", parSet_.getName().c_str())); }
          GenericToolbox::writeInTFile(saveDir_, errorsCanvas.get(), "fitConstraints");

        }; // makePrePostFitCompPlot

        makePrePostFitCompPlot(GenericToolbox::mkdirTFile(saveSubdir_, "values"), false);
        if( not GundamGlobals::isLightOutputMode() ) {
          makePrePostFitCompPlot(GenericToolbox::mkdirTFile(saveSubdir_, "valuesNorm"), true);
        }

      }; // savePostFitObjFct

  LogInfo << "Extracting post-fit errors..." << std::endl;
  for( const auto& parSet : getPropagator().getParametersManager().getParameterSetsList() ){
    if( not parSet.isEnabled() ){ continue; }

    LogInfo << "Extracting post-fit errors of parameter set: " << parSet.getName() << std::endl;
    auto* parSetDir = GenericToolbox::mkdirTFile(errorDir, parSet.getName());

    auto* parList = &parSet.getEffectiveParameterList();
    // dimension should be the right one -> parList includes the fixed one
    auto covMatrix = std::make_unique<TMatrixD>(int(parList->size()), int(parList->size()));
    for( auto& iPar : *parList ){
      int iMinimizerIndex = GenericToolbox::findElementIndex((Parameter*) &iPar, getMinimizerFitParameterPtr());
      if( iMinimizerIndex == -1 ) continue;
      for( auto& jPar : *parList ){
        int jMinimizerIndex = GenericToolbox::findElementIndex((Parameter*) &jPar, getMinimizerFitParameterPtr());
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
void MinimizerInterface::updateCacheToBestfitPoint(){
  LogThrowIf(_minimizer_->X() == nullptr, "No best fit point provided by the minimizer.");

  LogWarning << "Updating propagator cache to the best fit point..." << std::endl;
  getLikelihood().evalFit( _minimizer_->X() );
}

// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
