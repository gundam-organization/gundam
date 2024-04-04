//
// Created by Nadrino on 11/06/2021.
//

#include "FitterEngine.h"
#include "GundamGlobals.h"
#include "RootMinimizer.h"
#include "AdaptiveMcmc.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Json.h"
#include "GenericToolbox.Root.h"
#include "Logger.h"

#include <Math/Factory.h>
#include "TGraph.h"
#include "TLegend.h"

#include <cmath>
#include <memory>


LoggerInit([]{
  Logger::setUserHeaderStr("[FitterEngine]");
});


void FitterEngine::readConfigImpl(){
  LogInfo << "Reading FitterEngine config..." << std::endl;
  GenericToolbox::setT2kPalette();

  JsonType minimizerConfig{};
  std::string minimizerTypeStr{"RootMinimizer"};

  // legacy configs:
  GenericToolbox::Json::deprecatedAction(_config_, "mcmcConfig", [&]{
    LogAlert << "mcmcConfig should now be set as minimizerConfig" << std::endl;
    minimizerConfig = GenericToolbox::Json::fetchValue( _config_, "mcmcConfig" , minimizerConfig );
  });
  GenericToolbox::Json::deprecatedAction(_config_, "engineType", [&]{
    LogAlert << "engineType should now be specified withing minimizerConfig/minimizerType" << std::endl;
    minimizerTypeStr = GenericToolbox::Json::fetchValue( _config_, "engineType", minimizerTypeStr );

    // handle deprecated types
    if     ( minimizerTypeStr == "minimizer" ){ minimizerTypeStr = "RootMinimizer"; }
    else if( minimizerTypeStr == "mcmc" )     { minimizerTypeStr = "AdaptiveMCMC"; }
  });

  // new config format:
  minimizerConfig = GenericToolbox::Json::fetchValue( _config_, "minimizerConfig" , minimizerConfig );
  minimizerTypeStr = GenericToolbox::Json::fetchValue( minimizerConfig, "type", minimizerTypeStr );

  _minimizerType_ = MinimizerType::toEnum( minimizerTypeStr, true );
  switch( _minimizerType_.value ){
    case MinimizerType::RootMinimizer:
      this->_minimizer_ = std::make_unique<RootMinimizer>( this );
      break;
    case MinimizerType::AdaptiveMCMC:
      this->_minimizer_ = std::make_unique<AdaptiveMcmc>( this );
      break;
    default:
      LogThrow("Unknown minimizer type selected: " << minimizerTypeStr << std::endl << "Available: " << MinimizerType::generateEnumFieldsAsString());
  }

  // now the minimizer is created, forward deprecated options
  GenericToolbox::Json::deprecatedAction(_config_, "monitorRefreshRateInMs", [&]{
    LogAlert << "Forwarding the option to Propagator. Consider moving it into \"minimizerConfig:\"" << std::endl;
    _minimizer_->getMonitor().convergenceMonitor.setMaxRefreshRateInMs(GenericToolbox::Json::fetchValue<int>(_config_, "monitorRefreshRateInMs"));
  });
  GenericToolbox::Json::deprecatedAction(_config_, "propagatorConfig", [&]{
    LogAlert << R"("propagatorConfig" should now be set within "dataSetManagerConfig".)" << std::endl;
    // reading the config already since nested objects need to be filled up for handling further deprecation
    _likelihoodInterface_.getDataSetManager().getPropagator().setConfig( GenericToolbox::Json::fetchValue<JsonType>(_config_, "propagatorConfig") );
  });
  GenericToolbox::Json::deprecatedAction(_likelihoodInterface_.getDataSetManager().getPropagator().getConfig(), "scanConfig", [&]{
    LogAlert << R"("scanConfig" should now be set within "fitterEngineConfig".)" << std::endl;
    _parameterScanner_.setConfig( GenericToolbox::Json::fetchValue<JsonType>(_likelihoodInterface_.getDataSetManager().getPropagator().getConfig(), "scanConfig") );
  });

  _minimizer_->readConfig( minimizerConfig );
  _likelihoodInterface_.readConfig( GenericToolbox::Json::fetchValue(_config_, "likelihoodInterfaceConfig", _likelihoodInterface_.getConfig()) );
  _parameterScanner_.readConfig( GenericToolbox::Json::fetchValue(_config_, "scanConfig", _parameterScanner_.getConfig()) );
  LogInfo << "Convergence monitor will be refreshed every " << _minimizer_->getMonitor().convergenceMonitor.getMaxRefreshRateInMs() << "ms." << std::endl;


  // local config
  _enablePca_ = GenericToolbox::Json::fetchValue(_config_, {{"enablePca"}, {"runPcaCheck"}, {"fixGhostFitParameters"}}, _enablePca_);
  _pcaDeltaChi2Threshold_ = GenericToolbox::Json::fetchValue(_config_, {{"ghostParameterDeltaChi2Threshold"}, {"pcaDeltaChi2Threshold"}}, _pcaDeltaChi2Threshold_);

  _enablePreFitScan_ = GenericToolbox::Json::fetchValue(_config_, "enablePreFitScan", _enablePreFitScan_);
  _enablePostFitScan_ = GenericToolbox::Json::fetchValue(_config_, "enablePostFitScan", _enablePostFitScan_);
  _enablePreFitToPostFitLineScan_ = GenericToolbox::Json::fetchValue(_config_, "enablePreFitToPostFitLineScan", _enablePreFitToPostFitLineScan_);

  _generateSamplePlots_ = GenericToolbox::Json::fetchValue(_config_, "generateSamplePlots", _generateSamplePlots_);
  _generateOneSigmaPlots_ = GenericToolbox::Json::fetchValue(_config_, "generateOneSigmaPlots", _generateOneSigmaPlots_);

  _doAllParamVariations_ = GenericToolbox::Json::fetchValue(_config_, "enableParamVariations", _doAllParamVariations_);
  _allParamVariationsSigmas_ = GenericToolbox::Json::fetchValue(_config_, {{"paramVariationsSigmas"}, {"allParamVariations"}}, _allParamVariationsSigmas_);

  _scaleParStepWithChi2Response_ = GenericToolbox::Json::fetchValue(_config_, "scaleParStepWithChi2Response", _scaleParStepWithChi2Response_);
  _parStepGain_ = GenericToolbox::Json::fetchValue(_config_, "parStepGain", _parStepGain_);

  _throwMcBeforeFit_ = GenericToolbox::Json::fetchValue(_config_, "throwMcBeforeFit", _throwMcBeforeFit_);
  _throwGain_ = GenericToolbox::Json::fetchValue(_config_, "throwMcBeforeFitGain", _throwGain_);
  _savePostfitEventTrees_ = GenericToolbox::Json::fetchValue(_config_, "savePostfitEventTrees", _savePostfitEventTrees_);

  LogWarning << "FitterEngine configured." << std::endl;
}
void FitterEngine::initializeImpl(){
  LogThrowIf(_config_.empty(), "Config is not set.");
  LogThrowIf(_saveDir_== nullptr);

  if( GundamGlobals::isLightOutputMode() ){
    // TODO: this check should be more universal
    LogWarning << "Light mode enabled, wiping plot gen config..." << std::endl;
    _likelihoodInterface_.getDataSetManager().getPropagator().getPlotGenerator().readConfig(JsonType());
  }

  _likelihoodInterface_.initialize();

  _parameterScanner_.setLikelihoodInterfacePtr( &_likelihoodInterface_ );
  _parameterScanner_.initialize();

  if( _likelihoodInterface_.getDataSetManager().getPropagator().isThrowAsimovToyParameters() ){
    LogInfo << "Writing throws in TTree..." << std::endl;
    auto* throwsTree = new TTree("throws", "throws");

    std::vector<GenericToolbox::RawDataArray> thrownParameterValues{};
    thrownParameterValues.reserve(_likelihoodInterface_.getDataSetManager().getPropagator().getParametersManager().getParameterSetsList().size());
    for( auto& parSet : _likelihoodInterface_.getDataSetManager().getPropagator().getParametersManager().getParameterSetsList() ){
      if( not parSet.isEnabled() ) continue;

      std::vector<std::string> leavesList;
      thrownParameterValues.emplace_back();

      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() or par.isFixed() or par.isFree() ) continue;
        leavesList.emplace_back(GenericToolbox::generateCleanBranchName(par.getTitle()) + "/D");
        thrownParameterValues.back().writeRawData(par.getThrowValue());
      }

      thrownParameterValues.back().lockArraySize();
      throwsTree->Branch(
          GenericToolbox::generateCleanBranchName(parSet.getName()).c_str(),
          &thrownParameterValues.back().getRawDataArray()[0],
          GenericToolbox::joinVectorString(leavesList, ":").c_str()
      );
    }

    throwsTree->Fill();
    GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(_saveDir_, "preFit/parameters"), throwsTree);
  }

  // This moves the parameters
  if( _enablePca_ ) {
    LogWarning << "PCA is enabled. Polling parameters..." << std::endl;
    this->runPcaCheck();
  }

  // This moves the parameters
  if( _scaleParStepWithChi2Response_ ){
    LogInfo << "Using parameter step scale: " << _parStepGain_ << std::endl;
    this->rescaleParametersStepSize();
  }

  // The minimizer needs all the parameters to be fully setup (i.e. PCA done
  // and other properties)
  _minimizer_->initialize();

  if( GundamGlobals::getVerboseLevel() >= VerboseLevel::MORE_PRINTOUT ){ checkNumericalAccuracy(); }

  // Write data
  LogInfo << "Writing propagator objects..." << std::endl;
  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile(_saveDir_, "propagator"),
      TNamed("initialParameterState", GenericToolbox::Json::toReadableString(_likelihoodInterface_.getDataSetManager().getPropagator().getParametersManager().exportParameterInjectorConfig()).c_str())
  );

  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile(_saveDir_, "propagator"),
      _likelihoodInterface_.getDataSetManager().getPropagator().getParametersManager().getGlobalCovarianceMatrix().get(), "globalCovarianceMatrix"
  );
  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile(_saveDir_, "propagator"),
      _likelihoodInterface_.getDataSetManager().getPropagator().getParametersManager().getStrippedCovarianceMatrix().get(), "strippedCovarianceMatrix"
  );
  for( auto& parSet : _likelihoodInterface_.getDataSetManager().getPropagator().getParametersManager().getParameterSetsList() ){
    if(not parSet.isEnabled()) continue;

    auto saveFolder = GenericToolbox::joinPath( "propagator", parSet.getName() );
    GenericToolbox::writeInTFile(
        GenericToolbox::mkdirTFile( _saveDir_, saveFolder ),
        parSet.getPriorCovarianceMatrix().get(), "covarianceMatrix"
    );
    GenericToolbox::writeInTFile(
        GenericToolbox::mkdirTFile(_saveDir_, saveFolder ),
        parSet.getPriorCorrelationMatrix().get(), "correlationMatrix"
    );

    auto parsSaveFolder = GenericToolbox::joinPath( saveFolder, "parameters" );
    for( auto& par : parSet.getParameterList() ){
      auto parSaveFolder = GenericToolbox::joinPath(parsSaveFolder, GenericToolbox::generateCleanBranchName(par.getTitle()));
      auto outDir = GenericToolbox::mkdirTFile(_saveDir_, parSaveFolder );

      GenericToolbox::writeInTFile( outDir, TNamed( "title", par.getTitle().c_str() ) );
      GenericToolbox::writeInTFile( outDir, TNamed( "fullTitle", par.getFullTitle().c_str() ) );
      GenericToolbox::writeInTFile( outDir, TNamed( "name", par.getName().c_str() ) );
      GenericToolbox::writeInTFile( outDir, TNamed( "isEnabled", std::to_string( par.isEnabled() ).c_str() ) );
      GenericToolbox::writeInTFile( outDir, TNamed( "index", std::to_string( par.getParameterIndex() ).c_str() ) );
      GenericToolbox::writeInTFile( outDir, TNamed( "prior", std::to_string( par.getPriorValue() ).c_str() ) );
      GenericToolbox::writeInTFile( outDir, TNamed( "stdDev", std::to_string( par.getStdDevValue() ).c_str() ) );
      GenericToolbox::writeInTFile( outDir, TNamed( "priorType", std::to_string( par.getPriorType().value ).c_str() ) );
      GenericToolbox::writeInTFile( outDir, TNamed( "min", std::to_string( par.getMinValue() ).c_str() ) );
      GenericToolbox::writeInTFile( outDir, TNamed( "max", std::to_string( par.getMaxValue() ).c_str() ) );
    }

    if( parSet.isEnableEigenDecomp() ){
      auto eigenSaveFolder = GenericToolbox::joinPath( saveFolder, "eigen" );
      for( auto& eigen : parSet.getEigenParameterList() ){
        auto eigenFolder = GenericToolbox::joinPath(eigenSaveFolder, GenericToolbox::generateCleanBranchName(eigen.getTitle()));
        auto outDir = GenericToolbox::mkdirTFile( _saveDir_, eigenFolder );

        GenericToolbox::writeInTFile( outDir, TNamed( "title", eigen.getTitle().c_str() ) );
        GenericToolbox::writeInTFile( outDir, TNamed( "fullTitle", eigen.getFullTitle().c_str() ) );
        GenericToolbox::writeInTFile( outDir, TNamed( "name", eigen.getName().c_str() ) );
        GenericToolbox::writeInTFile( outDir, TNamed( "isEnabled", std::to_string( eigen.isEnabled() ).c_str() ) );
        GenericToolbox::writeInTFile( outDir, TNamed( "index", std::to_string( eigen.getParameterIndex() ).c_str() ) );
        GenericToolbox::writeInTFile( outDir, TNamed( "prior", std::to_string( eigen.getPriorValue() ).c_str() ) );
        GenericToolbox::writeInTFile( outDir, TNamed( "stdDev", std::to_string( eigen.getStdDevValue() ).c_str() ) );
        GenericToolbox::writeInTFile( outDir, TNamed( "priorType", std::to_string( eigen.getPriorType().value ).c_str() ) );
        GenericToolbox::writeInTFile( outDir, TNamed( "min", std::to_string( eigen.getMinValue() ).c_str() ) );
        GenericToolbox::writeInTFile( outDir, TNamed( "max", std::to_string( eigen.getMaxValue() ).c_str() ) );
      }
    }
  }


  if( _minimizerType_ == MinimizerType::RootMinimizer ){
    dynamic_cast<const RootMinimizer*>( &this->getMinimizer() )->saveMinimizerSettings( GenericToolbox::mkdirTFile(_saveDir_, "fit/minimizer" ) );
  }

  _likelihoodInterface_.propagateAndEvalLikelihood();

  if( not GundamGlobals::isLightOutputMode() ){
    _likelihoodInterface_.getDataSetManager().getTreeWriter().writeSamples(
        GenericToolbox::mkdirTFile(_saveDir_, "preFit/events"),
        _likelihoodInterface_.getDataSetManager().getPropagator()
    );
  }

  // writing event rates
  LogInfo << "Writing event rates..." << std::endl;
  for( auto& sample : _likelihoodInterface_.getDataSetManager().getPropagator().getSampleSet().getSampleList() ){
    if( not sample.isEnabled() ){ continue; }


    {
      TVectorD mcRate(1);
      mcRate[0] = sample.getMcContainer().getSumWeights();
      auto outDir = GenericToolbox::joinPath("preFit/rates", GenericToolbox::generateCleanBranchName(sample.getName()), "MC");
      GenericToolbox::writeInTFile(
          GenericToolbox::mkdirTFile(_saveDir_, outDir),
          mcRate, "sumWeights"
      );
    }

    {
      TVectorD dataRate(1);
      dataRate[0] = sample.getDataContainer().getSumWeights();
      auto outDir = GenericToolbox::joinPath("preFit/rates", GenericToolbox::generateCleanBranchName(sample.getName()), "Data");
      GenericToolbox::writeInTFile(
          GenericToolbox::mkdirTFile(_saveDir_, outDir),
          dataRate, "sumWeights"
      );
    }

  }


  LogWarning << "Saving all objects to disk..." << std::endl;
  GenericToolbox::triggerTFileWrite(_saveDir_);
}

// Core
void FitterEngine::fit(){
  LogWarning << __METHOD_NAME__ << std::endl;
  LogThrowIf( not isInitialized() );

  LogWarning << "Pre-fit likelihood state:" << std::endl;

  std::string llhState{_likelihoodInterface_.getSummary()};
  LogInfo << llhState << std::endl;
  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile( _saveDir_, "preFit" ),
      TNamed("llhState", llhState.c_str())
  );
  _preFitParState_ = _likelihoodInterface_.getDataSetManager().getPropagator().getParametersManager().exportParameterInjectorConfig();
  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile( _saveDir_, "preFit" ),
      TNamed("parState", GenericToolbox::Json::toReadableString(_preFitParState_).c_str())
  );

  // Not moving parameters
  if( _generateSamplePlots_ and not _likelihoodInterface_.getDataSetManager().getPropagator().getPlotGenerator().getConfig().empty() ){
    LogInfo << "Generating pre-fit sample plots..." << std::endl;
    _likelihoodInterface_.getDataSetManager().getPropagator().getPlotGenerator().generateSamplePlots(GenericToolbox::mkdirTFile(_saveDir_, "preFit/samples"));
    GenericToolbox::triggerTFileWrite(_saveDir_);
  }

  // Moving parameters
  if( _generateOneSigmaPlots_ and not _likelihoodInterface_.getDataSetManager().getPropagator().getPlotGenerator().getConfig().empty() ){
    LogInfo << "Generating pre-fit one-sigma variation plots..." << std::endl;
    _parameterScanner_.generateOneSigmaPlots(GenericToolbox::mkdirTFile(_saveDir_, "preFit/oneSigma"));
    GenericToolbox::triggerTFileWrite(_saveDir_);
  }
  if( _doAllParamVariations_ ){
    LogInfo << "Running all parameter variation on pre-fit samples..." << std::endl;
    _parameterScanner_.varyEvenRates(
        _allParamVariationsSigmas_,
        GenericToolbox::mkdirTFile(_saveDir_, "preFit/varyEventRates")
    );
    GenericToolbox::triggerTFileWrite(_saveDir_);
  }
  if( _enablePreFitScan_ ){
    LogInfo << "Scanning fit parameters before minimizing..." << std::endl;
    _minimizer_->scanParameters( GenericToolbox::mkdirTFile(_saveDir_, "preFit/scan") );
    GenericToolbox::triggerTFileWrite(_saveDir_);
  }
  if( _throwMcBeforeFit_ ){
    LogAlert << "Throwing correlated parameters of MC away from their prior..." << std::endl;
    LogAlert << "Throw gain form MC push set to: " << _throwGain_ << std::endl;
    for( auto& parSet : _likelihoodInterface_.getDataSetManager().getPropagator().getParametersManager().getParameterSetsList() ){
      if(not parSet.isEnabled()) continue;
      if( not parSet.isEnabledThrowToyParameters() ){
        LogWarning << "\"" << parSet.getName() << "\" has marked disabled throwMcBeforeFit: skipping." << std::endl;
        continue;
      }
      if( GenericToolbox::Json::doKeyExist(parSet.getConfig(), "customFitParThrow") ){

        LogAlert << "Using custom mc parameter push for " << parSet.getName() << std::endl;

        for(auto& entry : GenericToolbox::Json::fetchValue(parSet.getConfig(), "customFitParThrow", std::vector<JsonType>())){

          int parIndex = GenericToolbox::Json::fetchValue<int>(entry, "parIndex");

          auto& parList = parSet.getParameterList();
          double pushVal =
              parList[parIndex].getParameterValue()
              + parList[parIndex].getStdDevValue()
                * GenericToolbox::Json::fetchValue<double>(entry, "nbSigmaAway");

          LogWarning << "Pushing #" << parIndex << " to " << pushVal << std::endl;
          parList[parIndex].setParameterValue( pushVal );

          if( parSet.isEnableEigenDecomp() ){
            parSet.propagateOriginalToEigen();
          }

        }
        continue;
      }
      else{
        LogAlert << "Throwing correlated parameters for " << parSet.getName() << std::endl;
        parSet.throwParameters(true, _throwGain_);
      }
    } // parSet


    LogAlert << "Current LLH state:" << std::endl;
    _likelihoodInterface_.propagateAndEvalLikelihood();

    LogAlert << _likelihoodInterface_.getSummary() << std::endl;
  }

  // Leaving now?
  if( _isDryRun_ ){
    LogAlert << "Dry run requested. Leaving before the minimization." << std::endl;
    return;
  }

  LogInfo << "Minimizing LLH..." << std::endl;
  this->_minimizer_->minimize();

  LogWarning << "Saving post-fit par state..." << std::endl;
  _postFitParState_ = _likelihoodInterface_.getDataSetManager().getPropagator().getParametersManager().exportParameterInjectorConfig();
  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile( _saveDir_, "postFit" ),
      TNamed("parState", GenericToolbox::Json::toReadableString(_postFitParState_).c_str())
  );

  LogWarning << "Post-fit likelihood state:" << std::endl;
  llhState = _likelihoodInterface_.getSummary();
  LogInfo << llhState << std::endl;
  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile( _saveDir_, "postFit" ),
      TNamed("llhState", llhState.c_str())
  );


  if (_savePostfitEventTrees_){
    LogInfo << "Saving PostFit event Trees" << std::endl;
    _likelihoodInterface_.getDataSetManager().getTreeWriter().writeSamples(
        GenericToolbox::mkdirTFile(_saveDir_, "postFit/events"),
        _likelihoodInterface_.getDataSetManager().getPropagator()
    );
  }
  if( _generateSamplePlots_ and not _likelihoodInterface_.getDataSetManager().getPropagator().getPlotGenerator().getConfig().empty() ){
    LogInfo << "Generating post-fit sample plots..." << std::endl;
    _likelihoodInterface_.getDataSetManager().getPropagator().getPlotGenerator().generateSamplePlots(GenericToolbox::mkdirTFile(_saveDir_, "postFit/samples"));
    GenericToolbox::triggerTFileWrite(_saveDir_);
  }
  if( _enablePostFitScan_ ){
    LogInfo << "Scanning fit parameters around the minimum point..." << std::endl;
    _minimizer_->scanParameters( GenericToolbox::mkdirTFile(_saveDir_, "postFit/scan") );
    GenericToolbox::triggerTFileWrite(_saveDir_);
  }
  if( _enablePreFitToPostFitLineScan_ ){
    if( not GundamGlobals::isLightOutputMode() ){
      LogInfo << "Scanning along the line from pre-fit to post-fit points..." << std::endl;
      _parameterScanner_.scanSegment(GenericToolbox::mkdirTFile(_saveDir_, "postFit/scanConvergence"),
                                     _postFitParState_, _preFitParState_);
      GenericToolbox::triggerTFileWrite(_saveDir_);
    }
  }



  if( _minimizer_->getMinimizerStatus() != 0 ){
    LogError << "Skipping post-fit error calculation since the minimizer did not converge." << std::endl;
  }
  else{
    if( _minimizer_->isErrorCalcEnabled() ){
      LogInfo << "Computing post-fit errors..." << std::endl;
      _minimizer_->calcErrors();
    }
  }

  LogWarning << "Fit is done." << std::endl;
}

// protected
void FitterEngine::runPcaCheck(){

  _likelihoodInterface_.propagateAndEvalLikelihood();

  double baseLlh = _likelihoodInterface_.getLastLikelihood();
  double baseLlhStat = _likelihoodInterface_.getLastStatLikelihood();
  double baseLlhSyst = _likelihoodInterface_.getLastPenaltyLikelihood();

  LogInfo << "Reference stat log-likelihood for PCA: " << baseLlhStat << std::endl;

  // +1 sigma
  int iFitPar = -1;
  std::stringstream ssPrint;
  double deltaChi2Stat;

  for( auto& parSet : _likelihoodInterface_.getDataSetManager().getPropagator().getParametersManager().getParameterSetsList() ){

    if( not parSet.isEnabled() ){ continue; }

    if( not parSet.isEnablePca() ){
      LogWarning << "PCA disabled on " << parSet.getName() << ". Skipping..." << std::endl;
      continue;
    }
    else{
      LogInfo << "Performing PCA on " << parSet.getName() << "..." << std::endl;
    }

    bool fixNextEigenPars{false};
    auto& parList = parSet.getEffectiveParameterList();
    for( auto& par : parList ){
      LogScopeIndent;

      ssPrint.str("");
      ssPrint << "(" << par.getParameterIndex()+1 << "/" << parList.size() << ") +1 std-dev on " << parSet.getName() + "/" + par.getTitle();

      if( fixNextEigenPars ){
        par.setIsFixed(true);
#ifndef NOCOLOR
        std::string red(GenericToolbox::ColorCodes::redBackground);
        std::string rst(GenericToolbox::ColorCodes::resetColor);
#else
        std::string red;
        std::string rst;
#endif
//        LogInfo << red << ssPrint.str() << " -> FIXED AS NEXT EIGEN." << rst << std::endl;
        continue;
      }

      if( par.isEnabled() and not par.isFixed() ){
        double currentParValue = par.getParameterValue();
        par.setParameterValue( currentParValue + par.getStdDevValue() );

        ssPrint << " " << currentParValue << " -> " << par.getParameterValue();
        LogInfo << ssPrint.str() << "..." << std::endl;

        _likelihoodInterface_.propagateAndEvalLikelihood();

        deltaChi2Stat = _likelihoodInterface_.getLastStatLikelihood() - baseLlhStat;

        ssPrint << ": diff. stat log-likelihood = " << deltaChi2Stat;

        LogInfo.moveTerminalCursorBack(1);
        LogInfo << ssPrint.str() << std::endl;

        if( std::abs(deltaChi2Stat) < _pcaDeltaChi2Threshold_ ){
          par.setIsFixed(true); // ignored in the Chi2 computation of the parSet
          ssPrint << " < " << GenericToolbox::Json::fetchValue(_config_, {{"ghostParameterDeltaChi2Threshold"}, {"pcaDeltaChi2Threshold"}}, 1E-6) << " -> FIXED";
          LogInfo.moveTerminalCursorBack(1);
#ifndef NOCOLOR
          std::string red(GenericToolbox::ColorCodes::redBackground);
          std::string rst(GenericToolbox::ColorCodes::resetColor);
#else
          std::string red;
        std::string rst;
#endif
          LogInfo << red << ssPrint.str() << rst << std::endl;

          if( parSet.isEnableEigenDecomp() and GenericToolbox::Json::fetchValue(_config_, "fixGhostEigenParametersAfterFirstRejected", false) ){
            fixNextEigenPars = true;
          }
        }

        // Come back to the original values
        par.setParameterValue( currentParValue );
      }
    }

    // Recompute inverse matrix for the fitter.  Note: Eigen decomposed parSet
    // don't need a new inversion since the matrix is diagonal
    if( not parSet.isEnableEigenDecomp() ){
      parSet.processCovarianceMatrix();
    }

  }

  // comeback to old values
  _likelihoodInterface_.propagateAndEvalLikelihood();
}

void FitterEngine::rescaleParametersStepSize(){
  LogInfo << __METHOD_NAME__ << std::endl;

  _likelihoodInterface_.propagateAndEvalLikelihood();
  double baseLlhPull = _likelihoodInterface_.getLastPenaltyLikelihood();
  double baseLlh = _likelihoodInterface_.getLastLikelihood();

  // +1 sigma
  for( auto& parSet : _likelihoodInterface_.getDataSetManager().getPropagator().getParametersManager().getParameterSetsList() ){

    for( auto& par : parSet.getEffectiveParameterList() ){

      if( not par.isEnabled() ){ continue; }

      double currentParValue = par.getParameterValue();
      par.setParameterValue( currentParValue + par.getStdDevValue() );

      _likelihoodInterface_.propagateAndEvalLikelihood();

      double deltaChi2 = _likelihoodInterface_.getLastLikelihood() - baseLlh;
      double deltaChi2Pulls = _likelihoodInterface_.getLastPenaltyLikelihood() - baseLlhPull;

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
      _likelihoodInterface_.propagateAndEvalLikelihood();
      LogInfo << " -> Δχ²(step) = " << _likelihoodInterface_.getLastLikelihood() - baseLlh << std::endl;
      par.setParameterValue( currentParValue );
    }

  }

  _likelihoodInterface_.propagateAndEvalLikelihood();
}
void FitterEngine::checkNumericalAccuracy(){
  LogWarning << __METHOD_NAME__ << std::endl;
  int nTest{100}; int nThrows{10}; double gain{20};
  std::vector<std::vector<std::vector<double>>> throws(nThrows); // saved throws [throw][parSet][par]
  std::vector<double> responses(nThrows, std::nan("unset"));
  // stability/numerical accuracy test

  LogInfo << "Throwing..." << std::endl;
  for(auto& throwEntry : throws ){
    for( auto& parSet : _likelihoodInterface_.getDataSetManager().getPropagator().getParametersManager().getParameterSetsList() ){
      if(not parSet.isEnabled()) continue;
      if( not parSet.isEnabledThrowToyParameters() ){ continue;}
      parSet.throwParameters(true, gain);
      throwEntry.emplace_back(parSet.getParameterList().size(), 0);
      for( size_t iPar = 0 ; iPar < parSet.getParameterList().size() ; iPar++){
        throwEntry.back()[iPar] = parSet.getParameterList()[iPar].getParameterValue();
      }
      parSet.moveParametersToPrior();
    }
  }

  LogInfo << "Testing..." << std::endl;
  for( int iTest = 0 ; iTest < nTest ; iTest++ ){
    GenericToolbox::displayProgressBar(iTest, nTest, "Testing computational accuracy...");
    for( size_t iThrow = 0 ; iThrow < throws.size() ; iThrow++ ){
      int iParSet{-1};
      for( auto& parSet : _likelihoodInterface_.getDataSetManager().getPropagator().getParametersManager().getParameterSetsList() ){
        if(not parSet.isEnabled()) continue;
        if( not parSet.isEnabledThrowToyParameters() ){ continue;}
        iParSet++;
        for( size_t iPar = 0 ; iPar < parSet.getParameterList().size() ; iPar++){
          parSet.getParameterList()[iPar].setParameterValue( throws[iThrow][iParSet][iPar] );
        }
      }
      _likelihoodInterface_.getDataSetManager().getPropagator().propagateParameters();
      _likelihoodInterface_.evalLikelihood();

      if( responses[iThrow] == responses[iThrow] ){ // not nan
        LogThrowIf(_likelihoodInterface_.getLastLikelihood() != responses[iThrow], "Not accurate: " << _likelihoodInterface_.getLastLikelihood() - responses[iThrow] << " / "
                                                                                                    << GET_VAR_NAME_VALUE(_likelihoodInterface_.getLastLikelihood()) << " <=> " << GET_VAR_NAME_VALUE(responses[iThrow])
        )
      }
      responses[iThrow] = _likelihoodInterface_.getLastLikelihood();
    }
    LogDebug << GenericToolbox::toString(responses) << std::endl;
  }
  LogInfo << "OK" << std::endl;
}

