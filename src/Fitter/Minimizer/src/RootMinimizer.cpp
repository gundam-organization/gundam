//
// Created by Nadrino on 16/12/2021.
//

#include "LikelihoodInterface.h"
#include "RootMinimizer.h"
#include "FitterEngine.h"

#include "GundamGlobals.h"
#include "GundamUtils.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"

#include "Math/Factory.h"
#include "Math/Minimizer.h"
#include "Math/Functor.h"
#include "Fit/ParameterSettings.h"
#include "Minuit2/Minuit2Minimizer.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MinuitParameter.h"
#include "TLegend.h"


void RootMinimizer::configureImpl(){
  LogDebugIf(GundamGlobals::isDebug()) << "Configuring RootMinimizer..." << std::endl;

  // read general parameters first
  this->MinimizerBase::configureImpl();

  _config_.defineFields({
    {"monitorGradientDescent"},
    {"minimizer"},
    {"algorithm"},
    {"strategy"},
    {"print_level"},
    {"tolerance"},
    {"tolerancePerDegreeOfFreedom"},
    {"maxIterations", {"max_iter"}},
    {"maxFcnCalls", {"max_fcn"}},
    {"enableSimplexBeforeMinimize"},
    {"simplexMaxFcnCalls"},
    {"simplexToleranceLoose"},
    {"simplexStrategy"},
    {"errors", {"errorsAlgo"}},
    {"generatedPostFitParBreakdown"},
    {"generatedPostFitEigenBreakdown"},
    {"stepSizeScaling"},
    {"restoreStepSizeBeforeHesse"},
  });
  _config_.checkConfiguration();

  _config_.fillValue(gradientDescentMonitor.isEnabled, "monitorGradientDescent");
  _config_.fillValue(_minimizerType_, "minimizer");
  _config_.fillValue(_minimizerAlgo_, "algorithm");

  _config_.fillValue(_strategy_, "strategy");
  _config_.fillValue(_printLevel_, "print_level");
  _config_.fillValue(_tolerance_, "tolerance");
  _config_.fillValue(_tolerancePerDegreeOfFreedom_, "tolerancePerDegreeOfFreedom");
  _config_.fillValue(_maxIterations_, "maxIterations");
  _config_.fillValue(_maxFcnCalls_, "maxFcnCalls");

  _config_.fillValue(_preFitWithSimplex_, "enableSimplexBeforeMinimize");
  _config_.fillValue(_simplexMaxFcnCalls_, "simplexMaxFcnCalls");
  _config_.fillValue(_simplexToleranceLoose_, "simplexToleranceLoose");
  _config_.fillValue(_simplexStrategy_, "simplexStrategy");

  _config_.fillValue(_errorAlgo_, "errors");

  _config_.fillValue(_generatedPostFitParBreakdown_, "generatedPostFitParBreakdown");
  _config_.fillValue(_generatedPostFitEigenBreakdown_, "generatedPostFitEigenBreakdown");

  // old -- should flag as dev or deprecated?
  _config_.fillValue(_stepSizeScaling_, "stepSizeScaling");
  _config_.fillValue(_restoreStepSizeBeforeHesse_, "restoreStepSizeBeforeHesse");

}
void RootMinimizer::initializeImpl(){
  MinimizerBase::initializeImpl();

  if( gradientDescentMonitor.isEnabled ){
    _monitor_.convergenceMonitor.defineNewQuantity({ "LastStep", "Last step descent", [&](GenericToolbox::VariableMonitor& v){
      return GenericToolbox::parseUnitPrefix(gradientDescentMonitor.getLastStepDeltaValue(v.getName()), 8); }
    });
    _monitor_.convergenceMonitor.addDisplayedQuantity("LastStep");
    _monitor_.convergenceMonitor.getQuantity("LastStep").title = "Last step descent";

    gradientDescentMonitor.valueDefinitionList.emplace_back(
        "Total/dof", [](const RootMinimizer* this_){ return this_->getLikelihoodInterface().getLastLikelihood() / this_->fetchNbDegreeOfFreedom(); }
    );
    gradientDescentMonitor.valueDefinitionList.emplace_back(
        "Total", [](const RootMinimizer* this_){ return this_->getLikelihoodInterface().getBuffer().totalLikelihood; }
    );
    gradientDescentMonitor.valueDefinitionList.emplace_back(
        "Stat", [](const RootMinimizer* this_){ return this_->getLikelihoodInterface().getBuffer().statLikelihood; }
    );
    gradientDescentMonitor.valueDefinitionList.emplace_back(
        "Syst", [](const RootMinimizer* this_){ return this_->getLikelihoodInterface().getBuffer().penaltyLikelihood; }
    );
  }

  LogInfo << "Initializing RootMinimizer..." << std::endl;
  if( not std::isnan(_tolerancePerDegreeOfFreedom_) ) {
    LogWarning << "Using tolerance per degree of freedom: " << _tolerancePerDegreeOfFreedom_ << std::endl;
    _tolerance_ = _tolerancePerDegreeOfFreedom_ * fetchNbDegreeOfFreedom();
  }

  LogInfo << "Tolerance is set to: " << _tolerance_ << std::endl;
  LogInfo << "The minimizer will run until it reaches an Estimated Distance to Minimum (EDM) of: " << getTargetEdm() << std::endl;
  LogInfo << "EDM per degree of freedom is: " << getTargetEdm()/fetchNbDegreeOfFreedom() << std::endl;


  LogInfo << "Defining minimizer as: " << _minimizerType_ << "/" << _minimizerAlgo_ << std::endl;
  _rootMinimizer_ = std::unique_ptr<ROOT::Math::Minimizer>(
      ROOT::Math::Factory::CreateMinimizer(_minimizerType_, _minimizerAlgo_)
  );
  LogThrowIf(_rootMinimizer_ == nullptr, "Could not create minimizer: " << _minimizerType_ << "/" << _minimizerAlgo_);

  if( _minimizerAlgo_.empty() ){
    _minimizerAlgo_ = _rootMinimizer_->Options().MinimizerAlgorithm();
    LogWarning << "Using default minimizer algo: " << _minimizerAlgo_ << std::endl;
  }

  _functor_ = ROOT::Math::Functor(this, &RootMinimizer::evalFit, getMinimizerFitParameterPtr().size());
  _rootMinimizer_->SetFunction( _functor_ );
  _rootMinimizer_->SetStrategy(_strategy_);
  _rootMinimizer_->SetPrintLevel(_printLevel_);
  _rootMinimizer_->SetTolerance(_tolerance_);
  _rootMinimizer_->SetMaxIterations(_maxIterations_);
  _rootMinimizer_->SetMaxFunctionCalls(_maxFcnCalls_);

  for( std::size_t iFitPar = 0 ; iFitPar < getMinimizerFitParameterPtr().size() ; iFitPar++ ){
    auto& fitPar = *(getMinimizerFitParameterPtr()[iFitPar]);

    LogThrowIf(std::isnan(fitPar.getStepSize()), "No step size provided for: " << fitPar.getFullTitle());

    if( not useNormalizedFitSpace() ){
      _rootMinimizer_->SetVariable(iFitPar, fitPar.getFullTitle(), fitPar.getParameterValue(), fitPar.getStepSize() * _stepSizeScaling_);

      // strange ROOT parameter setting...
      if( fitPar.getParameterLimits().hasBothBounds() ){
        _rootMinimizer_->SetVariableLimits(iFitPar, fitPar.getParameterLimits().min, fitPar.getParameterLimits().max);
      }
      else if( fitPar.getParameterLimits().hasLowerBound() ){
        _rootMinimizer_->SetVariableLowerLimit(iFitPar, fitPar.getParameterLimits().min);
      }
      else if( fitPar.getParameterLimits().hasUpperBound() ){
        _rootMinimizer_->SetVariableUpperLimit(iFitPar, fitPar.getParameterLimits().max);
      }
    }
    else{
      _rootMinimizer_->SetVariable(iFitPar, fitPar.getFullTitle(),
                                   ParameterSet::toNormalizedParValue(fitPar.getParameterValue(), fitPar),
                                   ParameterSet::toNormalizedParRange(fitPar.getStepSize() * _stepSizeScaling_, fitPar)
      );
      // strange ROOT parameter setting...
      if( fitPar.getParameterLimits().hasBothBounds() ) {
        _rootMinimizer_->SetVariableLimits(
          iFitPar,
          ParameterSet::toNormalizedParValue(fitPar.getParameterLimits().min, fitPar),
          ParameterSet::toNormalizedParValue(fitPar.getParameterLimits().max, fitPar));
      }
      else if( fitPar.getParameterLimits().hasLowerBound() ){
        _rootMinimizer_->SetVariableLowerLimit(iFitPar, ParameterSet::toNormalizedParValue(fitPar.getParameterLimits().min, fitPar));
      }
      else if( fitPar.getParameterLimits().hasUpperBound() ){
        _rootMinimizer_->SetVariableUpperLimit(iFitPar, ParameterSet::toNormalizedParValue(fitPar.getParameterLimits().max, fitPar));
      }
    }
  }

  LogInfo << "RootMinimizer initialized." << std::endl;
}

void RootMinimizer::dumpFitParameterSettings() {
  LogInfo << "RootMinimizer fit parameters:" << std::endl;

  GenericToolbox::TablePrinter t;

  t << "#" << GenericToolbox::TablePrinter::NextColumn;
  t << "Name" << GenericToolbox::TablePrinter::NextColumn;
  t << "Value" << GenericToolbox::TablePrinter::NextColumn;
  t << "Step" << GenericToolbox::TablePrinter::NextColumn;
  t << "Fixed?" << GenericToolbox::TablePrinter::NextColumn;
  t << "Bounds" << GenericToolbox::TablePrinter::NextLine;

  for( std::size_t iFitPar = 0 ; iFitPar < getMinimizerFitParameterPtr().size() ; ++iFitPar ) {
    ROOT::Fit::ParameterSettings parSettings;
    _rootMinimizer_->GetVariableSettings(iFitPar,parSettings);

    t << iFitPar << GenericToolbox::TablePrinter::NextColumn;
    t << parSettings.Name() << GenericToolbox::TablePrinter::NextColumn;
    t << parSettings.Value() << GenericToolbox::TablePrinter::NextColumn;
    t << parSettings.StepSize() << GenericToolbox::TablePrinter::NextColumn;
    t << (parSettings.IsFixed() ? "Yes" : "No") << GenericToolbox::TablePrinter::NextColumn;

    GenericToolbox::Range r{std::nan("unset"), std::nan("unset")};
    if(parSettings.HasLowerLimit()) r.min = parSettings.LowerLimit();
    if(parSettings.HasUpperLimit()) r.max = parSettings.UpperLimit();

    t << r << GenericToolbox::TablePrinter::NextLine;
  }

  t.printTable();
}

void RootMinimizer::dumpMinuit2State() {
  auto* mn2 = dynamic_cast<ROOT::Minuit2::Minuit2Minimizer*>(_rootMinimizer_.get());
  if (not mn2) return;
  const ROOT::Minuit2::MnUserParameterState& mn2State = mn2->State();
  for( std::size_t iFitPar = 0 ;
       iFitPar < getMinimizerFitParameterPtr().size() ; ++iFitPar ) {
    const ROOT::Minuit2::MinuitParameter& par = mn2State.Parameter(iFitPar);
    LogDebug << "MINUIT2 #" << iFitPar;
    LogDebug << " Value: " << par.Value();
    LogDebug << " Fixed: " << par.IsFixed();
    if (par.HasLowerLimit()) {
      LogDebug << " Lower: " << par.LowerLimit();
    }
    if (par.HasUpperLimit()) {
      LogDebug << " Upper: " << par.UpperLimit();
    }
    LogDebug << " Name: " << par.GetName();
    LogDebug  << std::endl;
  }
}

void RootMinimizer::minimize(){
  // calling the common routine
  this->MinimizerBase::minimize();

  GenericToolbox::Time::Timer minimizationStopWatch;
  minimizationStopWatch.start();

  int nbFitCallOffset = getMonitor().nbEvalLikelihoodCalls;
  LogInfo << "Fit call offset: " << nbFitCallOffset << std::endl;

  /// Apply the frozen state to the root minimizer variable definitions
  for( std::size_t iFitPar = 0 ; iFitPar < getMinimizerFitParameterPtr().size() ; iFitPar++ ){
    auto& fitPar = *(getMinimizerFitParameterPtr()[iFitPar]);
    if (fitPar.isFrozen()) {
      if (fitPar.isEigen()) {
        LogAlert << "Eigen decomposed parameters cannot be frozen, and remain free"
                 << std::endl;
      }
      getMinimizer()->FixVariable(iFitPar);
    }
    else {
      getMinimizer()->ReleaseVariable(iFitPar);
    }
  }

  dumpFitParameterSettings();

  if( _preFitWithSimplex_ ){
    LogInfo << "Running simplex algo before the minimizer" << std::endl;
    LogThrowIf(_minimizerType_ != "Minuit2", "Can't launch simplex with " << _minimizerType_);

    std::string originalAlgo = _rootMinimizer_->Options().MinimizerAlgorithm();

    _rootMinimizer_->Options().SetMinimizerAlgorithm( "Simplex" );
    _rootMinimizer_->SetMaxFunctionCalls(_simplexMaxFcnCalls_);
    _rootMinimizer_->SetTolerance(_tolerance_ * _simplexToleranceLoose_ );
    _rootMinimizer_->SetStrategy(0);

    getMonitor().minimizerTitle = _minimizerType_ + "/" + "Simplex";
    getMonitor().stateTitleMonitor = "Running Simplex";
    getMonitor().stateTitleMonitor += " / Target EDM: " + std::to_string(getTargetEdm());

    // SIMPLEX
    getMonitor().isEnabled = true;
    // dumpFitParameterSettings(); // Dump internal ROOT::Minimizer info
    // dumpMinuit2State();         // Dump internal ROOT::Minuit2Minimizer info
    _fitHasConverged_ = _rootMinimizer_->Minimize();
    getMonitor().isEnabled = false;

    // Make sure we are on the right spot
    updateCacheToBestfitPoint();

    // export bf point with SIMPLEX
    LogInfo << "Writing " << _minimizerType_ << "/Simplex best fit parameters..." << std::endl;
    GenericToolbox::writeInTFileWithObjTypeExt(
        GenericToolbox::mkdirTFile( getOwner().getSaveDir(), GenericToolbox::joinPath("postFit", _minimizerAlgo_) ),
        TNamed("parameterStateAfterSimplex", GenericToolbox::Json::toReadableString( getModelPropagator().getParametersManager().exportParameterInjectorConfig() ).c_str() )
    );

    // Back to original
    _rootMinimizer_->Options().SetMinimizerAlgorithm(originalAlgo.c_str());
    _rootMinimizer_->SetMaxFunctionCalls(_maxFcnCalls_);
    _rootMinimizer_->SetTolerance(_tolerance_);
    _rootMinimizer_->SetStrategy(_strategy_);

    LogInfo << getMonitor().convergenceMonitor.generateMonitorString(); // lasting printout
    LogInfo << "Simplex ended after " << getMonitor().nbEvalLikelihoodCalls - nbFitCallOffset << " calls." << std::endl;
  }

  getMonitor().minimizerTitle = _minimizerType_ + "/" + _minimizerAlgo_;
  getMonitor().stateTitleMonitor = "Running " + _rootMinimizer_->Options().MinimizerAlgorithm();
  getMonitor().stateTitleMonitor += " / Target EDM: " + std::to_string(getTargetEdm());

  getMonitor().isEnabled = true;
  // dumpFitParameterSettings(); // Dump internal ROOT::Minimizer info
  // dumpMinuit2State();         // Dump internal ROOT::Minuit2Minimizer info
  _fitHasConverged_ = _rootMinimizer_->Minimize();
  _minimizeDone_ = true;
  getMonitor().isEnabled = false;

  minimizationStopWatch.stop();
  LogInfo << "Minimization stopped after " << GenericToolbox::toString(minimizationStopWatch.eval()) << std::endl;

  int nbMinimizeCalls = getMonitor().nbEvalLikelihoodCalls - nbFitCallOffset;

  LogInfo << getMonitor().convergenceMonitor.generateMonitorString(); // lasting printout
  LogInfo << "Minimization ended after " << nbMinimizeCalls << " calls." << std::endl;
  if(_minimizerType_ == "Minuit" or _minimizerType_ == "Minuit2") LogInfo << "Status code: " << GundamUtils::minuitStatusCodeStr.at(_rootMinimizer_->Status()) << std::endl;
  else LogInfo << "Status code: " << _rootMinimizer_->Status() << std::endl;
  if(_minimizerType_ == "Minuit" or _minimizerType_ == "Minuit2") LogInfo << "Covariance matrix status code: " << GundamUtils::covMatrixStatusCodeStr.at(_rootMinimizer_->CovMatrixStatus()) << std::endl;
  else LogInfo << "Covariance matrix status code: " << _rootMinimizer_->CovMatrixStatus() << std::endl;

  // Make sure we are on the right spot
  updateCacheToBestfitPoint();

  // export bf point
  LogInfo << "Writing " << _minimizerType_ << "/" << _minimizerAlgo_ << " best fit parameters..." << std::endl;
  GenericToolbox::writeInTFileWithObjTypeExt(
      GenericToolbox::mkdirTFile( getOwner().getSaveDir(), GenericToolbox::joinPath("postFit", _minimizerAlgo_) ),
      TNamed("parameterStateAfterMinimize", GenericToolbox::Json::toReadableString( getModelPropagator().getParametersManager().exportParameterInjectorConfig() ).c_str() )
  );

  if( getMonitor().historyTree != nullptr ){
    LogInfo << "Saving LLH history..." << std::endl;
    GenericToolbox::writeInTFileWithObjTypeExt(getOwner().getSaveDir(), getMonitor().historyTree.get());
  }

  if( gradientDescentMonitor.isEnabled ){ saveGradientSteps(); }

  if( _fitHasConverged_ ){ LogInfo << "Minimization has converged!" << std::endl; }
  else{ LogError << "Minimization did not converged." << std::endl; }

  LogInfo << "Post-fit event-rates:" << std::endl;
  LogInfo << getLikelihoodInterface().getSampleBreakdownTable() << std::endl;

  if( getOwner().getSaveDir() != nullptr ) {
    LogInfo << "Writing convergence stats..." << std::endl;
    int toyIndex = getModelPropagator().getIThrow();
    int nIterations = int(_rootMinimizer_->NIterations());
    int nFitPars = int(_rootMinimizer_->NFree());
    double edmBestFit = _rootMinimizer_->Edm();
    double fitStatus = _rootMinimizer_->Status();
    double covStatus = _rootMinimizer_->CovMatrixStatus();
    double chi2MinFitter = _rootMinimizer_->MinValue();
    double minimizationTimeInSec = minimizationStopWatch.eval().count();

    int nDof = fetchNbDegreeOfFreedom();
    int nbFitBins = getLikelihoodInterface().getNbSampleBins();

    auto bestFitStats = std::make_unique<TTree>("bestFitStats", "bestFitStats");
    bestFitStats->SetDirectory( nullptr );
    bestFitStats->Branch("fitConverged", &_fitHasConverged_);
    bestFitStats->Branch("fitStatusCode", &fitStatus);
    bestFitStats->Branch("covStatusCode", &covStatus);
    bestFitStats->Branch("edmBestFit", &edmBestFit);
    bestFitStats->Branch("nIterations", &nIterations);
    bestFitStats->Branch("chi2MinFitter", &chi2MinFitter);
    bestFitStats->Branch("toyIndex", &toyIndex);
    bestFitStats->Branch("nFitBins", &nbFitBins);
    bestFitStats->Branch("nbFreeParameters", getNbFreeParametersPtr());
    bestFitStats->Branch("nFitPars", &nFitPars);
    bestFitStats->Branch("nbDegreeOfFreedom", &nDof);

    bestFitStats->Branch("minimizationTimeInSec", &minimizationTimeInSec);
    bestFitStats->Branch("nCallsAtBestFit", &getMonitor().nbEvalLikelihoodCalls);
    bestFitStats->Branch("totalLikelihoodAtBestFit", &getLikelihoodInterface().getBuffer().totalLikelihood );
    bestFitStats->Branch("statLikelihoodAtBestFit",  &getLikelihoodInterface().getBuffer().statLikelihood );
    bestFitStats->Branch("penaltyLikelihoodAtBestFit",  &getLikelihoodInterface().getBuffer().penaltyLikelihood );

    std::vector<GenericToolbox::RawDataArray> samplesArrList(getModelPropagator().getSampleSet().getSampleList().size());
    int iSample{-1};
    for( auto& samplePair : getLikelihoodInterface().getSamplePairList() ){
      if( not samplePair.model->isEnabled() ) continue;

      std::vector<std::string> leavesDict;
      iSample++;

      leavesDict.emplace_back("llhSample/D");
      samplesArrList[iSample].writeRawData( getLikelihoodInterface().evalStatLikelihood( samplePair ) );

      int nBins = samplePair.model->getHistogram().getNbBins();
      for( int iBin = 0 ; iBin < nBins ; iBin++ ){
        leavesDict.emplace_back("llhSample_bin" + std::to_string(iBin) + "/D");
        samplesArrList[iSample].writeRawData( getLikelihoodInterface().getJointProbabilityPtr()->eval(samplePair, iBin) );
      }

      samplesArrList[iSample].lock();
      bestFitStats->Branch(
          GenericToolbox::generateCleanBranchName(samplePair.model->getName()).c_str(),
          &samplesArrList[iSample].getRawDataArray()[0],
          GenericToolbox::joinVectorString(leavesDict, ":").c_str()
      );
    }

    std::vector<GenericToolbox::RawDataArray> parameterSetArrList(getModelPropagator().getParametersManager().getParameterSetsList().size());
    int iParSet{-1};
    for( auto& parSet : getModelPropagator().getParametersManager().getParameterSetsList() ){
      if( not parSet.isEnabled() ) continue;

      std::vector<std::string> leavesDict;
      iParSet++;

      leavesDict.emplace_back("llhPenalty/D");
      parameterSetArrList[iParSet].writeRawData( getLikelihoodInterface().evalPenaltyLikelihood( parSet ) );

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
    GenericToolbox::mkdirTFile(getOwner().getSaveDir(), "postFit")->WriteObject(bestFitStats.get(), bestFitStats->GetName());

    LogInfo << "Writing " << _minimizerType_ << "/" << _minimizerAlgo_ << " post-fit errors" << std::endl;
    this->writePostFitData(GenericToolbox::mkdirTFile(getOwner().getSaveDir(), GenericToolbox::joinPath("postFit", _minimizerAlgo_)));
    GenericToolbox::triggerTFileWrite(GenericToolbox::mkdirTFile(getOwner().getSaveDir(), GenericToolbox::joinPath("postFit", _minimizerAlgo_)));

  }

  if( _fitHasConverged_ ){ setMinimizerStatus(0); }
  else{ setMinimizerStatus(_rootMinimizer_->Status()); }
}
void RootMinimizer::calcErrors(){

  LogThrowIf(not isInitialized(), "not initialized");

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Calling calcErrors()...") << std::endl;

  int nbFitCallOffset = getMonitor().nbEvalLikelihoodCalls;
  LogInfo << "Fit call offset: " << nbFitCallOffset << std::endl;

  if     ( _errorAlgo_ == "Minos" ){
    LogInfo << std::endl << GenericToolbox::addUpDownBars("Calling MINOS...") << std::endl;

    double errLow, errHigh;
    _rootMinimizer_->SetPrintLevel(0);

    for( int iFitPar = 0 ; iFitPar < _rootMinimizer_->NDim() ; iFitPar++ ){
      LogInfo << "Evaluating: " << _rootMinimizer_->VariableName(iFitPar) << "..." << std::endl;

      getMonitor().isEnabled = true;
      bool isOk = _rootMinimizer_->GetMinosError(iFitPar, errLow, errHigh);
      getMonitor().isEnabled = false;

#if ROOT_VERSION_CODE >= ROOT_VERSION(6,23,02)
      LogInfo << GundamUtils::minosStatusCodeStr.at(_rootMinimizer_->MinosStatus()) << std::endl;
#endif
      if( isOk ){
        LogInfo << _rootMinimizer_->VariableName(iFitPar) << ": " << errLow << " <- " << _rootMinimizer_->X()[iFitPar] << " -> +" << errHigh << std::endl;
      }
      else{
        LogError << _rootMinimizer_->VariableName(iFitPar) << ": " << errLow << " <- " << _rootMinimizer_->X()[iFitPar] << " -> +" << errHigh
                 << " - MINOS returned an error." << std::endl;
      }
    }

    // Put back at minimum
    for( int iFitPar = 0 ; iFitPar < _rootMinimizer_->NDim() ; iFitPar++ ){
      getMinimizerFitParameterPtr()[iFitPar]->setParameterValue(_rootMinimizer_->X()[iFitPar]);
    }
  } // Minos
  else if( _errorAlgo_ == "Hesse" ){

    if( _restoreStepSizeBeforeHesse_ ){
      LogWarning << "Restoring step size before HESSE..." << std::endl;
      for( int iFitPar = 0 ; iFitPar < _rootMinimizer_->NDim() ; iFitPar++ ){
        auto& par = *getMinimizerFitParameterPtr()[iFitPar];
        if(not useNormalizedFitSpace()){ _rootMinimizer_->SetVariableStepSize(iFitPar, par.getStepSize() * _stepSizeScaling_); }
        else{ _rootMinimizer_->SetVariableStepSize(iFitPar, ParameterSet::toNormalizedParRange(par.getStepSize() * _stepSizeScaling_, par)); } // should be 1
      }
    }

    // Make sure we are on the right spot
    updateCacheToBestfitPoint();

    getMonitor().minimizerTitle = _minimizerType_ + "/" + _errorAlgo_;
    getMonitor().stateTitleMonitor = "Running HESSE";

    GenericToolbox::Time::Timer errorStopWatch;
    errorStopWatch.start();

    getMonitor().isEnabled = true;
    _fitHasConverged_ = _rootMinimizer_->Hesse();
    getMonitor().isEnabled = false;

    errorStopWatch.stop();
    LogInfo << "Error calculation took: " << GenericToolbox::toString(errorStopWatch.eval()) << std::endl;

    LogInfo << "Hesse ended after " << getMonitor().nbEvalLikelihoodCalls - nbFitCallOffset << " calls." << std::endl;
    LogInfo << "HESSE status code: " << GundamUtils::hesseStatusCodeStr.at(_rootMinimizer_->Status()) << std::endl;
    LogInfo << "Covariance matrix status code: " << GundamUtils::covMatrixStatusCodeStr.at(_rootMinimizer_->CovMatrixStatus()) << std::endl;

    // Make sure we are on the right spot
    updateCacheToBestfitPoint();

    if(not _fitHasConverged_){
      LogError << "Hesse did not converge." << std::endl;
      LogError << getMonitor().convergenceMonitor.generateMonitorString(); // lasting printout
    }
    else{
      LogInfo << "Hesse converged." << std::endl;
      LogInfo << getMonitor().convergenceMonitor.generateMonitorString(); // lasting printout
    }

    int covStatus = _rootMinimizer_->CovMatrixStatus();

    auto hesseStats = std::make_unique<TTree>("hesseStats", "hesseStats");
    hesseStats->SetDirectory(nullptr);
    hesseStats->Branch("hesseSuccess", &_fitHasConverged_);
    hesseStats->Branch("covStatusCode", &covStatus);

    double errorTimeInSec = errorStopWatch.eval().count();
    hesseStats->Branch("errorTimeInSec", &errorTimeInSec);

    hesseStats->Fill();
    GenericToolbox::mkdirTFile( getOwner().getSaveDir(), "postFit")->WriteObject(hesseStats.get(), hesseStats->GetName());

    LogInfo << "Writing HESSE post-fit errors" << std::endl;
    this->writePostFitData(GenericToolbox::mkdirTFile(getOwner().getSaveDir(), "postFit/Hesse"));
    GenericToolbox::triggerTFileWrite(GenericToolbox::mkdirTFile(getOwner().getSaveDir(), "postFit/Hesse"));
  }
  else{
    LogError << GET_VAR_NAME_VALUE(_errorAlgo_) << " not implemented." << std::endl;
  }
}
void RootMinimizer::scanParameters( TDirectory* saveDir_ ){
  LogThrowIf(not isInitialized());

  getOwner().getParameterScanner().setGraphTitles( _minimizeDone_ ? "Post-fit scan": "Pre-fit scan" );

  LogInfo << "Performing scans of fit parameters..." << std::endl;
  for( int iPar = 0 ; iPar < getMinimizer()->NDim() ; iPar++ ){
    if( getMinimizer()->IsFixedVariable(iPar) ){
      LogWarning << getMinimizer()->VariableName(iPar)
                 << " is fixed. Skipping..." << std::endl;
      continue;
    }
    getOwner().getParameterScanner().scanParameter(*getMinimizerFitParameterPtr()[iPar], saveDir_);
  } // iPar
  for( auto& parSet : this->getModelPropagator().getParametersManager().getParameterSetsList() ){
    if( not parSet.isEnabled() ) continue;
    if( parSet.isEnableEigenDecomp() ){
      LogWarning << parSet.getName() << " is using eigen decomposition. Scanning original parameters..." << std::endl;
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ) continue;
        getOwner().getParameterScanner().scanParameter(par, saveDir_);
      }
    }
  }

  GenericToolbox::triggerTFileWrite(saveDir_);
}
double RootMinimizer::evalFit(const double *parArray_){
  auto out = MinimizerBase::evalFit(parArray_);

  // check the gradient steps
  if( GenericToolbox::toLowerCase(_minimizerType_) == "minuit2" ){

    if( gradientDescentMonitor.isEnabled ){

      auto& gradient = gradientDescentMonitor;

      // When gradient descent base minimizer probe a point toward the
      // minimum, every parameter get updated
      size_t nbValidPars = std::count_if(
              _minimizerParameterPtrList_.begin(), _minimizerParameterPtrList_.end(),
              [](const Parameter* par_){ return not ( par_->isFixed() or not par_->isEnabled() ); } );
      size_t nParUpdated = std::count_if(
              _minimizerParameterPtrList_.begin(), _minimizerParameterPtrList_.end(),
              [](const Parameter* par_){ return par_->gotUpdated(); } );

      bool isGradientDescentStep = (nParUpdated == nbValidPars);

      if( nParUpdated >= 5 ) {
        // It's a partial gradient step (should be more than 5 since HESSE can change 4 params at a time)
        // Some parameters can be left unchanged by the minimizer: could indicate some problems in the parametrization
        isGradientDescentStep = true;
      }

      if( isGradientDescentStep or gradient.stepPointList.empty() ){

        if( gradient.stepPointList.empty() ){
          // add the initial point
          LogWarning << "Adding initial point of the gradient monitor: ";
          gradient.addStep( this );
        }
        else{
          if( gradient.stepPointList.back().fitCallNb == _monitor_.nbEvalLikelihoodCalls - 1 ){
            LogWarning << "Minimizer is adjusting the step size: ";
            gradient.fillLastStep( this );
          }
          else{
            LogWarning << "Gradient step detected at iteration #" << _monitor_.nbEvalLikelihoodCalls;
            if(nParUpdated != nbValidPars){ LogWarning << " (PARTIAL: " << nParUpdated << "/" << nbValidPars << ")"; }
            LogWarning << ": ";
            gradient.addStep( this );
          }
        }

        if( gradient.stepPointList.size() >= 2 ){
          LogWarning << gradient.getLastStepValue("Total") + gradient.getLastStepDeltaValue("Total") << " -> ";
        }
        LogWarning << gradient.getLastStepValue("Total") << std::endl;
      }
    }
  }

  return out;
}


// const getters
double RootMinimizer::getTargetEdm() const{
  // Migrad: The default tolerance is 0.1, and the minimization will stop
  // when the estimated vertical distance to the minimum (EDM) is less
  // than 0.001*[tolerance]*UP (see SET ERR).
  // UP:
  // Minuit defines parameter errors as the change in parameter value required
  // to change the function value by UP. Normally, for chisquared fits
  // UP=1, and for negative log likelihood, UP=0.5
  return 0.001 * _tolerance_ * 1;
}

// core
void RootMinimizer::saveMinimizerSettings( TDirectory* saveDir_) const {
  LogInfo << "Saving minimizer settings..." << std::endl;

  GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("minimizerType", _minimizerType_.c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("minimizerAlgo", _minimizerAlgo_.c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("strategy", std::to_string(_strategy_).c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("printLevel", std::to_string(_printLevel_).c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("targetEDM", std::to_string(this->getTargetEdm()).c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("maxIterations", std::to_string(_maxIterations_).c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("maxFcnCalls", std::to_string(_maxFcnCalls_).c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("tolerance", std::to_string(_tolerance_).c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("stepSizeScaling", std::to_string(_stepSizeScaling_).c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("useNormalizedFitSpace", std::to_string(useNormalizedFitSpace()).c_str()) );

  if( _preFitWithSimplex_ ){
    GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("enableSimplexBeforeMinimize", std::to_string(_preFitWithSimplex_).c_str()) );
    GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("simplexMaxFcnCalls", std::to_string(_simplexMaxFcnCalls_).c_str()) );
    GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("simplexToleranceLoose", std::to_string(_simplexToleranceLoose_).c_str()) );
    GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("simplexStrategy", std::to_string(_simplexStrategy_).c_str()) );
  }

  if( isErrorCalcEnabled() ){
    GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("enablePostFitErrorFit", std::to_string(isErrorCalcEnabled()).c_str()) );
    GenericToolbox::writeInTFileWithObjTypeExt( saveDir_, TNamed("errorAlgo", _errorAlgo_.c_str()) );
  }
}

// protected
void RootMinimizer::writePostFitData( TDirectory* saveDir_) {
  LogInfo << __METHOD_NAME__ << std::endl;
  LogThrowIf(not isInitialized(), "not initialized");
  LogThrowIf(saveDir_==nullptr, "Save dir not specified");

  LogInfo << "Extracting post-fit covariance matrix" << std::endl;
  auto* matricesDir = GenericToolbox::mkdirTFile(saveDir_, "hessian");

  TMatrixDSym postfitCovarianceMatrix(int(_rootMinimizer_->NDim()));
  TMatrixDSym postfitHessianMatrix(int(_rootMinimizer_->NDim()));
  _rootMinimizer_->GetCovMatrix(postfitCovarianceMatrix.GetMatrixArray());
  _rootMinimizer_->GetCovMatrix(postfitHessianMatrix.GetMatrixArray());

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
      for( int iPar = 0 ; iPar < _rootMinimizer_->NDim() ; iPar++ ){
        hist_->GetXaxis()->SetBinLabel(iPar+1, _rootMinimizer_->VariableName(iPar).c_str());
        if(hist_->GetDimension() >= 2) hist_->GetYaxis()->SetBinLabel(iPar+1, _rootMinimizer_->VariableName(iPar).c_str());
      }
      applyLooks(hist_);
    };

    {
      LogInfo << "Writing post-fit cov matrices" << std::endl;
      auto postFitCovarianceTH2D = std::unique_ptr<TH2D>(GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) &postfitCovarianceMatrix) );
      applyBinLabels(postFitCovarianceTH2D.get());
      GenericToolbox::writeInTFileWithObjTypeExt(outDir_, postFitCovarianceTH2D.get(), "postfitCovariance");

      auto postfitCorrelationMatrix = std::unique_ptr<TMatrixD>(GenericToolbox::convertToCorrelationMatrix((TMatrixD*) &postfitCovarianceMatrix));
      auto postfitCorrelationTH2D = std::unique_ptr<TH2D>(GenericToolbox::convertTMatrixDtoTH2D(postfitCorrelationMatrix.get()));
      applyBinLabels(postfitCorrelationTH2D.get());
      postfitCorrelationTH2D->GetZaxis()->SetRangeUser(-1,1);
      GenericToolbox::writeInTFileWithObjTypeExt(outDir_, postfitCorrelationTH2D.get(), "postfitCorrelation");
    }

    {
      LogInfo << "Writing post-fit hessian matrices" << std::endl;
      auto postFitHessianTH2D = std::unique_ptr<TH2D>(GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) &postfitHessianMatrix) );
      applyBinLabels(postFitHessianTH2D.get());
      GenericToolbox::writeInTFileWithObjTypeExt(outDir_, postFitHessianTH2D.get(), "postfitHessian");

      auto postfitHessianCorMatrix = std::unique_ptr<TMatrixD>(GenericToolbox::convertToCorrelationMatrix((TMatrixD*) &postfitHessianMatrix));
      auto postfitHessianCorTH2D = std::unique_ptr<TH2D>(GenericToolbox::convertTMatrixDtoTH2D(postfitHessianCorMatrix.get()));
      applyBinLabels(postfitHessianCorTH2D.get());
      postfitHessianCorTH2D->GetZaxis()->SetRangeUser(-1,1);
      GenericToolbox::writeInTFileWithObjTypeExt(outDir_, postfitHessianCorTH2D.get(), "postfitHessianCorrelation");
    }

    // Fitter covariance matrix decomposition
    // first check if the post-cov matrix has NaN -> TMatrixDSymEigen crashes otherwise
    bool hasNan{false};
    for( int iRow = 0 ; iRow < postfitCovarianceMatrix.GetNrows() ; iRow++ ){
      for( int iCol = 0 ; iCol < postfitCovarianceMatrix.GetNcols() ; iCol++ ){
        if( std::isnan(postfitCovarianceMatrix[iRow][iCol]) ){ hasNan = true; break; }
      }
    }


    if( hasNan ){ LogAlert << "Skipping cov matrix decomposition as NaN values are present." << std::endl; }
    else{
      LogInfo << "Eigen decomposition of the post-fit covariance matrix" << std::endl;
      TMatrixDSymEigen decompCovMatrix(postfitCovarianceMatrix);

      auto eigenVectors = std::unique_ptr<TH2D>( GenericToolbox::convertTMatrixDtoTH2D(&decompCovMatrix.GetEigenVectors()) );
      applyBinLabels(eigenVectors.get());

      if( not GundamGlobals::isLightOutputMode() ) {
        GenericToolbox::writeInTFileWithObjTypeExt(GenericToolbox::mkdirTFile(outDir_, "eigenDecomposition"), eigenVectors.get(), "eigenVectors");
      }

      auto eigenValues = std::unique_ptr<TH1D>( GenericToolbox::convertTVectorDtoTH1D(&decompCovMatrix.GetEigenValues()) );

      applyBinLabels(eigenValues.get());
      if( not GundamGlobals::isLightOutputMode() ) {
        GenericToolbox::writeInTFileWithObjTypeExt(GenericToolbox::mkdirTFile(outDir_, "eigenDecomposition"), eigenValues.get(), "eigenValues");
      }

      double conditioning = decompCovMatrix.GetEigenValues().Min() / decompCovMatrix.GetEigenValues().Max();
      LogInfo << "Post-fit error conditioning is: " << conditioning << std::endl;

      LogInfo << "Reconstructing postfit hessian matrix..." << std::endl;
      auto eigenValuesInv = TVectorD(decompCovMatrix.GetEigenValues());
      for( int iEigen = 0 ; iEigen < eigenValuesInv.GetNrows() ; iEigen++ ){ eigenValuesInv[iEigen] = 1./eigenValuesInv[iEigen]; }
      auto diagonalMatrixInv = std::unique_ptr<TMatrixD>( GenericToolbox::makeDiagonalMatrix(&eigenValuesInv) );
      auto invEigVectors = TMatrixD(decompCovMatrix.GetEigenVectors());
      invEigVectors.T();

      TMatrixD hessianMatrix(int(_rootMinimizer_->NDim()), int(_rootMinimizer_->NDim())); hessianMatrix.Zero();
      hessianMatrix += decompCovMatrix.GetEigenVectors();
      hessianMatrix *= (*diagonalMatrixInv);
      hessianMatrix *= invEigVectors;

      TH2D* postfitHessianTH2D = GenericToolbox::convertTMatrixDtoTH2D(&hessianMatrix);
      applyBinLabels(postfitHessianTH2D);
      if( not GundamGlobals::isLightOutputMode() ){
        GenericToolbox::writeInTFileWithObjTypeExt(outDir_, postfitHessianTH2D, "postfitHessianReconstructed");
      }

      if( _generatedPostFitEigenBreakdown_ ){
        LogInfo << "Eigen breakdown..." << std::endl;
        TH1D eigenBreakdownHist("eigenBreakdownHist", "eigenBreakdownHist",
                                int(_rootMinimizer_->NDim()), -0.5, int(_rootMinimizer_->NDim()) - 0.5);
        std::vector<TH1D> eigenBreakdownAccum(decompCovMatrix.GetEigenValues().GetNrows(), eigenBreakdownHist);
        TH1D* lastAccumHist{nullptr};
        std::string progressTitle = LogWarning.getPrefixString() + "Accumulating eigen components...";
        for (int iEigen = decompCovMatrix.GetEigenValues().GetNrows() - 1; iEigen >= 0; iEigen--) {
          GenericToolbox::displayProgressBar(decompCovMatrix.GetEigenValues().GetNrows() - iEigen, decompCovMatrix.GetEigenValues().GetNrows(), progressTitle);
          // iEigen = 0 -> largest error contribution
          // Drawing in the back -> iEigen = 0 should be last in the accum plot
          if( lastAccumHist != nullptr ) eigenBreakdownAccum[iEigen] = *lastAccumHist;
          else eigenBreakdownAccum[iEigen] = eigenBreakdownHist;
          lastAccumHist = &eigenBreakdownAccum[iEigen];

          eigenBreakdownHist.SetTitle(Form("Parameter breakdown for eigen #%i = %f", iEigen,
                                           decompCovMatrix.GetEigenValues()[iEigen]));
          eigenBreakdownHist.SetLineColor(GenericToolbox::defaultColorWheel[iEigen%int(GenericToolbox::defaultColorWheel.size())]);
          eigenBreakdownHist.SetLabelSize(0.02);
          for ( int iPar = int(_rootMinimizer_->NDim()) - 1; iPar >= 0; iPar--) {
            eigenBreakdownHist.SetBinContent(iPar + 1,
                                             decompCovMatrix.GetEigenVectors()[iPar][iEigen] *
                                             decompCovMatrix.GetEigenVectors()[iPar][iEigen] *
                                             decompCovMatrix.GetEigenValues()[iEigen]
            );
          }
          applyBinLabels(&eigenBreakdownHist);
          applyBinLabels(&eigenBreakdownAccum[iEigen]);

          GenericToolbox::writeInTFileWithObjTypeExt(
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
          GenericToolbox::cleanupForDisplay(&accumPlot);
          GenericToolbox::writeInTFileWithObjTypeExt(GenericToolbox::mkdirTFile(outDir_, "eigenDecomposition"), &accumPlot, "eigenBreakdown");
        }
      }

      if( _generatedPostFitParBreakdown_ ){
        LogInfo << "Parameters breakdown..." << std::endl;
        TH1D parBreakdownHist("parBreakdownHist", "parBreakdownHist",
                              decompCovMatrix.GetEigenValues().GetNrows(), -0.5,
                              decompCovMatrix.GetEigenValues().GetNrows() - 0.5);
        std::vector<TH1D> parBreakdownAccum(_rootMinimizer_->NDim());
        TH1D* lastAccumHist{nullptr};
        for ( int iPar = int(_rootMinimizer_->NDim()) - 1; iPar >= 0; iPar--){

          if( lastAccumHist != nullptr ) parBreakdownAccum[iPar] = *lastAccumHist;
          else parBreakdownAccum[iPar] = parBreakdownHist;
          lastAccumHist = &parBreakdownAccum[iPar];

          parBreakdownHist.SetLineColor(GenericToolbox::defaultColorWheel[iPar%int(GenericToolbox::defaultColorWheel.size())]);

          parBreakdownHist.SetTitle(Form("Eigen breakdown for parameter #%i: %s", iPar, _rootMinimizer_->VariableName(iPar).c_str()));
          for (int iEigen = decompCovMatrix.GetEigenValues().GetNrows() - 1; iEigen >= 0; iEigen--){
            parBreakdownHist.SetBinContent(
                iPar+1,
                decompCovMatrix.GetEigenVectors()[iPar][iEigen]
                * decompCovMatrix.GetEigenVectors()[iPar][iEigen]
                * decompCovMatrix.GetEigenValues()[iEigen]
            );
          }
          GenericToolbox::writeInTFileWithObjTypeExt(
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
          GenericToolbox::cleanupForDisplay(&accumPlot);
          GenericToolbox::writeInTFileWithObjTypeExt(
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
      for( const auto& parSet : getModelPropagator().getParametersManager().getParameterSetsList() ){ if( parSet.isEnabled() ) nGlobalPars += int(parSet.getNbParameters()); }

      // Reconstruct the global passage matrix
      std::vector<std::string> parameterLabels(nGlobalPars);
      auto globalPassageMatrix = std::make_unique<TMatrixD>(nGlobalPars, nGlobalPars);
      for(int i = 0 ; i < nGlobalPars; i++ ){ (*globalPassageMatrix)[i][i] = 1; }
      int blocOffset{0};
      for( const auto& parSet : getModelPropagator().getParametersManager().getParameterSetsList() ){
        if( not parSet.isEnabled() ) continue;

        auto* parList = &parSet.getParameterList(); // we want the original names
        for( auto& par : *parList ){ parameterLabels[blocOffset + par.getParameterIndex()] = par.getFullTitle(); }

        parList = &parSet.getEffectiveParameterList();
        if( parSet.isEnableEigenDecomp() ){
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
      for( const auto& iParSet : getModelPropagator().getParametersManager().getParameterSetsList() ){
        if( not iParSet.isEnabled() ) continue;

        auto* iParList = &iParSet.getEffectiveParameterList();
        for( auto& iPar : *iParList ){
          int iMinimizerIndex = GenericToolbox::findElementIndex((Parameter*) &iPar, getMinimizerFitParameterPtr());

          int jOffset{0};
          for( const auto& jParSet : getModelPropagator().getParametersManager().getParameterSetsList() ){
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
                if( &iParSet == &jParSet and iParSet.isEnableEigenDecomp() ){
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
      GenericToolbox::writeInTFileWithObjTypeExt(outDir_, postfitCovarianceOriginalTH2D, "postfitCovarianceOriginal");

      TH2D* postfitCorrelationOriginalTH2D = GenericToolbox::convertTMatrixDtoTH2D(GenericToolbox::convertToCorrelationMatrix(originalCovMatrix.get()));
      applyBinLabelsOrig(postfitCorrelationOriginalTH2D, parameterNonFixedLabels);
      postfitCorrelationOriginalTH2D->GetZaxis()->SetRangeUser(-1,1);
      GenericToolbox::writeInTFileWithObjTypeExt(outDir_, postfitCorrelationOriginalTH2D, "postfitCorrelationOriginal");
    }

  };

  if( useNormalizedFitSpace() ){
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

        GenericToolbox::writeInTFileWithObjTypeExt(GenericToolbox::mkdirTFile(saveSubdir_, "matrices"), covMatrix_, "Covariance");
        GenericToolbox::writeInTFileWithObjTypeExt(GenericToolbox::mkdirTFile(saveSubdir_, "matrices"), covMatrixTH2D, "Covariance");
        GenericToolbox::writeInTFileWithObjTypeExt(GenericToolbox::mkdirTFile(saveSubdir_, "matrices"), corMatrix, "Correlation");
        GenericToolbox::writeInTFileWithObjTypeExt(GenericToolbox::mkdirTFile(saveSubdir_, "matrices"), corMatrixTH2D, "Correlation");
        GenericToolbox::writeInTFileWithObjTypeExt(GenericToolbox::mkdirTFile(saveSubdir_, "matrices"), corMatrix, "Correlation");

        GenericToolbox::cleanupForDisplay(corMatrixCanvas.get());
        GenericToolbox::writeInTFileWithObjTypeExt(GenericToolbox::mkdirTFile(saveSubdir_, "matrices"), corMatrixCanvas.get(), "Correlation");

        // Table printout
        GenericToolbox::TablePrinter t;

        t << "Parameter" << GenericToolbox::TablePrinter::NextColumn;
        t << "Prior Value" << GenericToolbox::TablePrinter::NextColumn;
        t << "Fit Value" << GenericToolbox::TablePrinter::NextColumn;
        t << "Diff Value" << GenericToolbox::TablePrinter::NextColumn;
        t << "Prior Err" << GenericToolbox::TablePrinter::NextColumn;
        t << "Fit Err" << GenericToolbox::TablePrinter::NextColumn;
        t << "Prior Fraction" << GenericToolbox::TablePrinter::NextLine;

        for( const auto& par : parList_ ){
          if( par.isEnabled() and not par.isFixed() ){
            double priorFraction = std::sqrt((*covMatrix_)[par.getParameterIndex()][par.getParameterIndex()]) / par.getStdDevValue();
            std::stringstream ss;
#ifndef NOCOLOR
            std::string red(GenericToolbox::ColorCodes::redBackground);
            std::string ylw(GenericToolbox::ColorCodes::yellowBackground);
            std::string blu(GenericToolbox::ColorCodes::blueLightText);
            std::string rst(GenericToolbox::ColorCodes::resetColor);
#else
            std::string red;
            std::string ylw;
            std::string blu;
            std::string rst;
#endif

            if( priorFraction < 1E-2 ){ t.setColorBuffer(ylw); }
            if( priorFraction > 1 ){ t.setColorBuffer(red); }
            if( par.isFree() ){ t.setColorBuffer(blu); }

            t << par.getFullTitle() << GenericToolbox::TablePrinter::NextColumn;
            t << par.getPriorValue() << GenericToolbox::TablePrinter::NextColumn;
            t << par.getParameterValue() << GenericToolbox::TablePrinter::NextColumn;
            t << par.getParameterValue() - par.getPriorValue() << GenericToolbox::TablePrinter::NextColumn;
            t << par.getStdDevValue() << GenericToolbox::TablePrinter::NextColumn;
            t << std::sqrt((*covMatrix_)[par.getParameterIndex()][par.getParameterIndex()]) << GenericToolbox::TablePrinter::NextColumn;

            std::string colorStr;
            if( par.isFree() ){ t << "Unconstrained"; }
            else{ t << priorFraction*100 << R"( %)"; }

            t << GenericToolbox::TablePrinter::NextLine;
            t.setColorBuffer(rst);
          }
        }
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
              if (par.isEnabled()) {
                postFitErrorHist->SetBinContent( 1 + par.getParameterIndex(),
                                                 par.getParameterValue());
                postFitErrorHist->SetBinError(
                  1 + par.getParameterIndex(),
                  std::sqrt((*covMatrix_)
                              [par.getParameterIndex()]
                              [par.getParameterIndex()]));
              }
              preFitErrorHist->SetBinContent( 1 + par.getParameterIndex(), par.getPriorValue() );
              if( par.isEnabled() and not par.isFixed() and not par.isFree() ){
                preFitErrorHist->SetBinError( 1 + par.getParameterIndex(), par.getStdDevValue() );
              }
            }
            else{
              if (par.isEnabled()) {
                postFitErrorHist->SetBinContent(
                  1 + par.getParameterIndex(),
                  ParameterSet::toNormalizedParValue(par.getParameterValue(),
                                                     par));
                postFitErrorHist->SetBinError(
                  1 + par.getParameterIndex(),
                  ParameterSet::toNormalizedParRange(
                    std::sqrt((*covMatrix_)
                                [par.getParameterIndex()]
                                [par.getParameterIndex()]), par));
              }

              preFitErrorHist->SetBinContent( 1 + par.getParameterIndex(), 0 );
              if( par.isEnabled() and not par.isFixed() and not par.isFree() ){
                preFitErrorHist->SetBinError( 1 + par.getParameterIndex(), 1 );
              }
            } // norm
          } // par

          if( getLikelihoodInterface().getDataType() == LikelihoodInterface::DataType::Toy ){
            bool draw{false};

            for( auto& par : parList_ ){
              double val{par.getThrowValue()};

              if( not std::isnan(val) ){ draw = true; }
              else{ val = par.getPriorValue(); }

              if( isNorm_ ){ val = ParameterSet::toNormalizedParValue(val, par);}
              toyParametersLine->SetBinContent(1+par.getParameterIndex(), val);
            }

            // don't draw the throw line if none is valid
            if( not draw ){ toyParametersLine = nullptr;}
          }

          auto yBounds = GenericToolbox::getYBounds({preFitErrorHist.get(), postFitErrorHist.get(), toyParametersLine.get()});

          for( const auto& par : parList_ ){
            TBox b(preFitErrorHist->GetBinLowEdge(1+par.getParameterIndex()), yBounds.min,
                   preFitErrorHist->GetBinLowEdge(1+par.getParameterIndex()+1), yBounds.max);
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
          preFitErrorHist->GetYaxis()->SetRangeUser(yBounds.min, yBounds.max);
          GenericToolbox::writeInTFileWithObjTypeExt(saveDir_, preFitErrorHist.get());

          postFitErrorHist->SetLineColor(9);
          postFitErrorHist->SetLineWidth(2);
          postFitErrorHist->SetMarkerColor(9);
          postFitErrorHist->SetMarkerStyle(kFullDotLarge);
          postFitErrorHist->SetTitle(Form("Post-fit Errors of %s", parSet_.getName().c_str()));
          GenericToolbox::writeInTFileWithObjTypeExt(saveDir_, postFitErrorHist.get());

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
          GenericToolbox::cleanupForDisplay(errorsCanvas.get());
          GenericToolbox::writeInTFileWithObjTypeExt(saveDir_, errorsCanvas.get(), "fitConstraints");

        }; // makePrePostFitCompPlot

        makePrePostFitCompPlot(GenericToolbox::mkdirTFile(saveSubdir_, "values"), false);
        if( not GundamGlobals::isLightOutputMode() ) {
          makePrePostFitCompPlot(GenericToolbox::mkdirTFile(saveSubdir_, "valuesNorm"), true);
        }

      }; // savePostFitObjFct

  LogInfo << "Extracting post-fit errors..." << std::endl;
  for( const auto& parSet : getModelPropagator().getParametersManager().getParameterSetsList() ){
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
    if( parSet.isEnableEigenDecomp() ){
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
void RootMinimizer::updateCacheToBestfitPoint(){
  LogThrowIf(_rootMinimizer_ == nullptr, "Invalid root minimizer");
  if (_rootMinimizer_->X() == nullptr) {
    LogError << "Minimizer error with "
             << _rootMinimizer_->Options().MinimizerType()
             << ":" << _rootMinimizer_->Options().MinimizerAlgorithm()
             << std::endl;
    LogThrow("No best fit point provided by the minimizer.");
  }

  LogInfo << "Updating propagator cache to the best fit point..." << std::endl;
  this->evalFit(_rootMinimizer_->X() );
}
void RootMinimizer::saveGradientSteps(){

  if( GundamGlobals::isLightOutputMode() ){
    LogAlert << "Skipping saveGradientSteps as light output mode is fired." << std::endl;
    return;
  }

  LogInfo << "Saving " << gradientDescentMonitor.stepPointList.size() << " gradient steps..." << std::endl;

  int nPointsPerStep{8};
  int maxNbPoints{1000}; // don't spend too much time eval the steps
  nPointsPerStep = std::min(nPointsPerStep, maxNbPoints/int(gradientDescentMonitor.stepPointList.size()));
  if( nPointsPerStep <= 1 ){ nPointsPerStep = 2; }

  // make sure the parameter states get restored as we leave
  auto currentParState = getModelPropagator().getParametersManager().exportParameterInjectorConfig();
  GenericToolbox::ScopedGuard g{
      [&](){
        ParametersManager::muteLogger();
        ParameterSet::muteLogger();
        ParameterScanner::muteLogger();
      },
      [&](){
        getModelPropagator().getParametersManager().injectParameterValues( currentParState );
        ParametersManager::unmuteLogger();
        ParameterSet::unmuteLogger();
        ParameterScanner::unmuteLogger();
      }
  };

  // load starting point
  auto lastParStep{getOwner().getPreFitParState()};

  std::vector<ParameterScanner::GraphEntry> globalGraphList;
  for(size_t iGradStep = 0 ; iGradStep < gradientDescentMonitor.stepPointList.size() ; iGradStep++ ){
    GenericToolbox::displayProgressBar(iGradStep, gradientDescentMonitor.stepPointList.size(), LogInfo.getPrefixString() + "Saving gradient steps...");

    // why do we need to remute the logger at each loop??
    ParameterSet::muteLogger(); Propagator::muteLogger(); ParametersManager::muteLogger();
    getModelPropagator().getParametersManager().injectParameterValues(gradientDescentMonitor.stepPointList[iGradStep].parState );

    getLikelihoodInterface().propagateAndEvalLikelihood();

    if( not GundamGlobals::isLightOutputMode() and gradientDescentMonitor.writeGradientSteps ) {
      auto outDir = GenericToolbox::mkdirTFile(getOwner().getSaveDir(), Form("fit/gradient/steps/step_%i", int(iGradStep)));
      GenericToolbox::writeInTFileWithObjTypeExt(outDir, TNamed("parState", GenericToolbox::Json::toReadableString(gradientDescentMonitor.stepPointList[iGradStep].parState).c_str()));
      GenericToolbox::writeInTFileWithObjTypeExt(outDir, TNamed("llhState", getLikelihoodInterface().getSummary().c_str()));
    }

    // line scan from previous point
    getParameterScanner().scanSegment( nullptr, gradientDescentMonitor.stepPointList[iGradStep].parState, lastParStep, nPointsPerStep );
    lastParStep = gradientDescentMonitor.stepPointList[iGradStep].parState;

    if( globalGraphList.empty() ){
      // copy
      globalGraphList = getParameterScanner().getGraphEntriesBuf();
    }
    else{
      // current
      auto& grEntries = getParameterScanner().getGraphEntriesBuf();

      for( size_t iEntry = 0 ; iEntry < globalGraphList.size() ; iEntry++ ){
        for(int iPt = 0 ; iPt < grEntries[iEntry].graph.GetN() ; iPt++ ){
          globalGraphList[iEntry].graph.AddPoint( grEntries[iEntry].graph.GetX()[iPt], grEntries[iEntry].graph.GetY()[iPt] );
        }
      }

    }
  }

  if( not globalGraphList.empty() ){
    for( auto& gEntry : globalGraphList ){
      gEntry.scanDataPtr->title = "Minimizer path to minimum";
      if( gradientDescentMonitor.writeDescentPaths ) {
        ParameterScanner::writeGraphEntry(gEntry, GenericToolbox::mkdirTFile(getOwner().getSaveDir(), "fit/gradient/global") );
      }
    }

    if( gradientDescentMonitor.writeDescentPathsRelative ) {
      auto* outDir = GenericToolbox::mkdirTFile(getOwner().getSaveDir(), "fit/gradient/globalRelative");
      for( auto& gEntry : globalGraphList ){
        if( gEntry.graph.GetN() == 0 ){ continue; }

        double minY{gEntry.graph.GetY()[gEntry.graph.GetN()-1]};
        double maxY{gEntry.graph.GetY()[0]};
        double delta{1E-6*std::abs( maxY - minY )};
        // allow log scale
        minY += delta;

        for( int iPt = 0 ; iPt < gEntry.graph.GetN() ; iPt++ ){
          gEntry.graph.GetY()[iPt] -= minY;
        }
        gEntry.scanDataPtr->title = "Minimizer path to minimum (difference)";
        ParameterScanner::writeGraphEntry(gEntry, outDir);
      }
    }

  }

}

// Local Variables:
// mode:c++
// c-basic-offset:2
// End:
