//
// Created by Nadrino on 11/06/2021.
//

#include "FitterEngine.h"
#include "GundamGlobals.h"
#include "RootMinimizer.h"
#include "SimpleMcmc.h"

#ifdef GUNDAM_USING_CACHE_MANAGER
#include "CacheManager.h"
#endif

#include "GenericToolbox.Utils.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"

#include "GundamAlmostEqual.h"

#include <cmath>
#include <memory>


void FitterEngine::configureImpl(){
  LogInfo << "Reading FitterEngine config..." << std::endl;
  GenericToolbox::setT2kPalette();

  // setting up the minimizer
  ConfigReader minimizerConfig{};
  _config_.fillValue(minimizerConfig, {{"minimizerConfig"}, {"mcmcConfig"}});

  std::string minimizerTypeStr{"RootMinimizer"};
  _config_.deprecatedAction("engineType", "minimizerConfig/type", [&]{
    _config_.fillValue(minimizerTypeStr, "engineType");

    // handle deprecated types
    if     ( minimizerTypeStr == "minimizer" ){ minimizerTypeStr = "RootMinimizer"; }
    else if( minimizerTypeStr == "mcmc" )     { minimizerTypeStr = "SimpleMCMC"; }
  });
  minimizerConfig.fillValue(minimizerTypeStr, "type");

  _minimizerType_ = MinimizerType::toEnum( minimizerTypeStr, true );
  switch( _minimizerType_.value ){
    case MinimizerType::RootMinimizer:
      this->_minimizer_ = std::make_unique<RootMinimizer>( this );
      break;
    case MinimizerType::SimpleMCMC:
      this->_minimizer_ = std::make_unique<SimpleMcmc>( this );
      break;
    default:
      LogExit("Unknown minimizer type selected: " << minimizerTypeStr << std::endl << "Available: " << MinimizerType::generateEnumFieldsAsString());
  }

  // now the minimizer is created, forward deprecated options
  _config_.deprecatedAction("monitorRefreshRateInMs", "fitterEngineConfig/minimizerConfig", [&]{
    _minimizer_->getMonitor().convergenceMonitor.setMaxRefreshRateInMs(_config_.fetchValue<int>("monitorRefreshRateInMs"));
  });
  _minimizer_->configure( minimizerConfig );

  _config_.deprecatedAction("propagatorConfig", "fitterEngineConfig/likelihoodInterfaceConfig", [&]{
    // reading the config already since nested objects need to be filled up for handling further deprecation
    getLikelihoodInterface().getModelPropagator().setConfig( _config_.fetchValue<ConfigReader>("propagatorConfig") );
  });
  _config_.fillValue(_likelihoodInterface_.getConfig(), "likelihoodInterfaceConfig");
  _likelihoodInterface_.configure();

  getLikelihoodInterface().getModelPropagator().getConfig().deprecatedAction("scanConfig", "fitterEngineConfig", [&]{
    _parameterScanner_.setConfig( getLikelihoodInterface().getModelPropagator().getConfig().fetchValue<ConfigReader>("scanConfig") );
  });
  _config_.fillValue(_parameterScanner_.getConfig(), {{"parameterScannerConfig"},{"scanConfig"}});
  _parameterScanner_.configure();

  LogInfo << "Convergence monitor will be refreshed every " << _minimizer_->getMonitor().convergenceMonitor.getMaxRefreshRateInMs() << "ms." << std::endl;

  // local config
  _config_.fillValue(_enablePca_, {{"enablePca"},{"runPcaCheck"},{"fixGhostFitParameters"}});

  _config_.fillEnum(_pcaMethod_, "pcaMethod");
  _config_.fillValue(_pcaThreshold_, {{"pcaThreshold"},{"pcaDeltaLlhThreshold"},{"pcaDeltaChi2Threshold"},{"ghostParameterDeltaChi2Threshold"}});

  _config_.fillValue(_enablePreFitScan_, "enablePreFitScan");
  _config_.fillValue(_enablePostFitScan_, "enablePostFitScan");
  _config_.fillValue(_enablePreFitToPostFitLineScan_, "enablePreFitToPostFitLineScan");

  _config_.fillValue(_generateSamplePlots_, "generateSamplePlots");
  _config_.fillValue(_generateOneSigmaPlots_, "generateOneSigmaPlots");

  _config_.fillValue(_doAllParamVariations_, "enableParamVariations");
  _config_.fillValue(_allParamVariationsSigmas_, {{"paramVariationsSigmas"},{"allParamVariations"}});

  _config_.fillValue(_scaleParStepWithChi2Response_, "scaleParStepWithChi2Response");
  _config_.fillValue(_parStepGain_, "parStepGain");

  _config_.fillValue(_throwMcBeforeFit_, "throwMcBeforeFit");
  _config_.fillValue(_throwGain_, "throwMcBeforeFitGain");
  _config_.fillValue(_savePostfitEventTrees_, "savePostfitEventTrees");

  LogInfo << "FitterEngine configured." << std::endl;
}
void FitterEngine::initializeImpl(){

  _config_.printUnusedKeys();

  if( GundamGlobals::isLightOutputMode() ){
    // TODO: this check should be more universal
    LogWarning << "Light mode enabled, wiping plot gen config..." << std::endl;
    getLikelihoodInterface().getPlotGenerator().configure(ConfigReader());
  }

  getLikelihoodInterface().initialize();

  _parameterScanner_.setLikelihoodInterfacePtr( &getLikelihoodInterface() );
  _parameterScanner_.initialize();

  if( _likelihoodInterface_.getDataType() == LikelihoodInterface::DataType::Toy ){
    LogInfo << "Writing throws in TTree..." << std::endl;
    auto* throwsTree = new TTree("throws", "throws");

    std::vector<GenericToolbox::RawDataArray> thrownParameterValues{};
    thrownParameterValues.reserve(getLikelihoodInterface().getModelPropagator().getParametersManager().getParameterSetsList().size());
    for( auto& parSet : getLikelihoodInterface().getModelPropagator().getParametersManager().getParameterSetsList() ){
      if( not parSet.isEnabled() ) continue;

      std::vector<std::string> leavesList;
      thrownParameterValues.emplace_back();

      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() or par.isFixed() or par.isFree() ) continue;
        leavesList.emplace_back(GenericToolbox::generateCleanBranchName(par.getTitle()) + "/D");
        thrownParameterValues.back().writeRawData(par.getThrowValue());
      }

      thrownParameterValues.back().lock();
      throwsTree->Branch(
          GenericToolbox::generateCleanBranchName(parSet.getName()).c_str(),
          &thrownParameterValues.back().getRawDataArray()[0],
          GenericToolbox::joinVectorString(leavesList, ":").c_str()
      );
    }

    throwsTree->Fill();
    GenericToolbox::writeInTFileWithObjTypeExt(GenericToolbox::mkdirTFile(_saveDir_, "preFit/parameters"), throwsTree);
  }

  // This moves the parameters
  if( _enablePca_ ) {
    LogAlert << "PCA is enabled. Polling parameters..." << std::endl;
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

  // Write data
  LogInfo << "Writing propagator objects..." << std::endl;
  GenericToolbox::writeInTFileWithObjTypeExt(
      GenericToolbox::mkdirTFile(_saveDir_, "propagator"),
      TNamed("initialParameterState", GenericToolbox::Json::toReadableString(getLikelihoodInterface().getModelPropagator().getParametersManager().exportParameterInjectorConfig()).c_str())
  );

  GenericToolbox::writeInTFileWithObjTypeExt(
      GenericToolbox::mkdirTFile(_saveDir_, "propagator"),
      getLikelihoodInterface().getModelPropagator().getParametersManager().getGlobalCovarianceMatrix().get(), "globalCovarianceMatrix"
  );
  GenericToolbox::writeInTFileWithObjTypeExt(
      GenericToolbox::mkdirTFile(_saveDir_, "propagator"),
      getLikelihoodInterface().getModelPropagator().getParametersManager().getStrippedCovarianceMatrix().get(), "strippedCovarianceMatrix"
  );
  for( auto& parSet : getLikelihoodInterface().getModelPropagator().getParametersManager().getParameterSetsList() ){
    if(not parSet.isEnabled()) continue;

    auto saveFolder = GenericToolbox::joinPath( "propagator", parSet.getName() );
    GenericToolbox::writeInTFileWithObjTypeExt(
        GenericToolbox::mkdirTFile( _saveDir_, saveFolder ),
        parSet.getPriorCovarianceMatrix().get(), "covarianceMatrix"
    );
    GenericToolbox::writeInTFileWithObjTypeExt(
        GenericToolbox::mkdirTFile(_saveDir_, saveFolder ),
        GenericToolbox::toCorrelationMatrix(parSet.getPriorCovarianceMatrix().get()),
        "correlationMatrix"
    );
    GenericToolbox::writeInTFileWithObjTypeExt(
        GenericToolbox::mkdirTFile( _saveDir_, saveFolder ),
        parSet.getInverseCovarianceMatrix().get(), "invCovarianceMatrix"
    );

    auto parsSaveFolder = GenericToolbox::joinPath( saveFolder, "parameters" );
    for( auto& par : parSet.getParameterList() ){
      auto parSaveFolder = GenericToolbox::joinPath(parsSaveFolder, GenericToolbox::generateCleanBranchName(par.getTitle()));
      auto outDir = GenericToolbox::mkdirTFile(_saveDir_, parSaveFolder );

      GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "title", par.getTitle().c_str() ) );
      GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "fullTitle", par.getFullTitle().c_str() ) );
      GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "name", par.getName().c_str() ) );
      GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "isEnabled", std::to_string( par.isEnabled() ).c_str() ) );
      GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "index", std::to_string( par.getParameterIndex() ).c_str() ) );
      GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "prior", std::to_string( par.getPriorValue() ).c_str() ) );
      GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "stdDev", std::to_string( par.getStdDevValue() ).c_str() ) );
      GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "priorType", std::to_string( par.getPriorType().value ).c_str() ) );
      GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "min", std::to_string( par.getParameterLimits().min ).c_str() ) );
      GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "max", std::to_string( par.getParameterLimits().max ).c_str() ) );
    }

    if( parSet.isEnableEigenDecomp() ){
      auto eigenSaveFolder = GenericToolbox::joinPath( saveFolder, "eigen" );
      for( auto& eigen : parSet.getEigenParameterList() ){
        auto eigenFolder = GenericToolbox::joinPath(eigenSaveFolder, GenericToolbox::generateCleanBranchName(eigen.getTitle()));
        auto outDir = GenericToolbox::mkdirTFile( _saveDir_, eigenFolder );

        GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "title", eigen.getTitle().c_str() ) );
        GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "fullTitle", eigen.getFullTitle().c_str() ) );
        GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "name", eigen.getName().c_str() ) );
        GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "isEnabled", std::to_string( eigen.isEnabled() ).c_str() ) );
        GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "index", std::to_string( eigen.getParameterIndex() ).c_str() ) );
        GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "prior", std::to_string( eigen.getPriorValue() ).c_str() ) );
        GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "stdDev", std::to_string( eigen.getStdDevValue() ).c_str() ) );
        GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "priorType", std::to_string( eigen.getPriorType().value ).c_str() ) );
        GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "min", std::to_string( eigen.getParameterLimits().min ).c_str() ) );
        GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed( "max", std::to_string( eigen.getParameterLimits().max ).c_str() ) );
      }
    }
  }


  if( _minimizerType_ == MinimizerType::RootMinimizer ){
    dynamic_cast<const RootMinimizer*>( &this->getMinimizer() )->saveMinimizerSettings( GenericToolbox::mkdirTFile(_saveDir_, "fit/minimizer" ) );
  }

  getLikelihoodInterface().propagateAndEvalLikelihood();

  LogInfo << "Writing sample histograms..." << std::endl;
  auto writeSampleHistFct = [&](const Sample* samplePtr_, const std::string& tag_){
    auto hist = std::make_unique<TH1D>(
          "histogram",
          Form("%s sample \"%s\"", tag_.c_str(), samplePtr_->getName().c_str()),
          samplePtr_->getHistogram().getNbBins(),
          0, samplePtr_->getHistogram().getNbBins()
        );
    for( int iBin = 1 ; iBin < hist->GetNbinsX()+1 ; iBin++ ){
      hist->SetBinContent( iBin, samplePtr_->getHistogram().getBinContentList()[iBin-1].sumWeights );
      hist->SetBinError( iBin, samplePtr_->getHistogram().getBinContentList()[iBin-1].sqrtSumSqWeights );
    }
    hist->GetXaxis()->SetTitle("Bin index");
    hist->GetYaxis()->SetTitle("Rate");
    hist->SetLineWidth(3);
    GenericToolbox::writeInTFile(
      GenericToolbox::TFilePath(_saveDir_, "preFit").getSubDir(tag_).getSubDir(samplePtr_->getName()).getDir(),
      hist.get()
    );
  };
  for( auto& samplePair : getLikelihoodInterface().getSamplePairList() ){
    writeSampleHistFct(samplePair.model, "model");
    writeSampleHistFct(samplePair.data, "data");
  }

  if( not GundamGlobals::isLightOutputMode() ){
    getLikelihoodInterface().writeEvents( GenericToolbox::TFilePath(_saveDir_, "preFit") );
    getLikelihoodInterface().writeEventRates( GenericToolbox::TFilePath(_saveDir_, "preFit") );
  }

  LogInfo << "Saving all objects to disk..." << std::endl;
  GenericToolbox::triggerTFileWrite(_saveDir_);
}

// Core
void FitterEngine::fit(){
  LogInfo << __METHOD_NAME__ << std::endl;
  LogThrowIf( not isInitialized() );

  LogInfo << "Pre-fit likelihood state:" << std::endl;

  auto preFitDir(GenericToolbox::mkdirTFile( _saveDir_, "preFit" ));

  std::string llhState{getLikelihoodInterface().getSummary()};
  LogInfo << llhState << std::endl;
  GenericToolbox::writeInTFileWithObjTypeExt(preFitDir, llhState, "llhState");
  _preFitParState_ = getLikelihoodInterface().getModelPropagator().getParametersManager().exportParameterInjectorConfig();
  GenericToolbox::writeInTFileWithObjTypeExt(preFitDir, GenericToolbox::Json::toReadableString(_preFitParState_), "parState");

  // Not moving parameters
  if( _generateSamplePlots_ and not getLikelihoodInterface().getPlotGenerator().getConfig().empty() ){
    LogInfo << "Generating pre-fit sample plots..." << std::endl;
    getLikelihoodInterface().getPlotGenerator().generateSamplePlots(GenericToolbox::mkdirTFile(_saveDir_, "preFit/plots"));
    GenericToolbox::triggerTFileWrite(_saveDir_);
  }

  // Moving parameters
  if( _generateOneSigmaPlots_ and not getLikelihoodInterface().getPlotGenerator().getConfig().empty() ){
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
    for( auto& parSet : getLikelihoodInterface().getModelPropagator().getParametersManager().getParameterSetsList() ){
      if(not parSet.isEnabled()) continue;
      if( not parSet.isEnabledThrowToyParameters() ){
        LogWarning << "\"" << parSet.getName() << "\" has marked disabled throwMcBeforeFit: skipping." << std::endl;
        continue;
      }
      if( parSet.getConfig().hasKey("customFitParThrow") ){

        LogAlert << "Using custom mc parameter push for " << parSet.getName() << std::endl;

        for(auto& entry : parSet.getConfig().fetchValue("customFitParThrow", std::vector<JsonType>())){

          int parIndex = GenericToolbox::Json::fetchValue<int>(entry, "parIndex");

          auto& parList = parSet.getParameterList();
          double pushVal =
              parList[parIndex].getParameterValue()
              + parList[parIndex].getStdDevValue()
                * GenericToolbox::Json::fetchValue<double>(entry, "nbSigmaAway");

          LogAlert << "Pushing #" << parIndex << " to " << pushVal << std::endl;
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


    LogInfo << "Current LLH state:" << std::endl;
    getLikelihoodInterface().propagateAndEvalLikelihood();

    LogInfo << getLikelihoodInterface().getSummary() << std::endl;
  }

  // Leaving now?
  if( _isDryRun_ ){
    LogAlert << "Dry run requested. Leaving before the minimization." << std::endl;
    return;
  }

#ifdef GUNDAM_USING_CACHE_MANAGER
  // To calculate the llh, we only need to grab the bin content, not
  // individual weight
  bool origCopy = Cache::Manager::SetIsEventWeightCopyEnabled( false );
#endif

  LogInfo << "Minimizing LLH..." << std::endl;
  this->_minimizer_->minimize();

  // re-evaluating since the minimizer might not have triggered an eval of the
  // LLH.  And guarantee that the weights were copied if that was enabled.
  getLikelihoodInterface().propagateAndEvalLikelihood();

#ifdef GUNDAM_USING_CACHE_MANAGER
  if( Cache::Manager::IsBuilt() ){
    LogInfo << "Copy Cache::Manager weights after minimizer" << std::endl;
    Cache::Manager::Get()->CopyEventWeights();
  }
  Cache::Manager::SetIsEventWeightCopyEnabled(origCopy);
#endif

  LogInfo << "Saving post-fit par state..." << std::endl;
  _postFitParState_ = getLikelihoodInterface().getModelPropagator().getParametersManager().exportParameterInjectorConfig();
  GenericToolbox::writeInTFileWithObjTypeExt(
      GenericToolbox::mkdirTFile( _saveDir_, "postFit" ),
      TNamed("parState", GenericToolbox::Json::toReadableString(_postFitParState_).c_str())
  );

  LogInfo << "Post-fit likelihood state:" << std::endl;
  llhState = getLikelihoodInterface().getSummary();
  LogInfo << llhState << std::endl;
  GenericToolbox::writeInTFileWithObjTypeExt(
      GenericToolbox::mkdirTFile( _saveDir_, "postFit" ),
      TNamed("llhState", llhState.c_str())
  );


  if (_savePostfitEventTrees_){
    if( not GundamGlobals::isLightOutputMode() ){
      LogInfo << "Saving PostFit event Trees" << std::endl;
      getLikelihoodInterface().writeEvents( GenericToolbox::TFilePath(_saveDir_, "postFit") );
    }
  }
  if( _generateSamplePlots_ and not getLikelihoodInterface().getPlotGenerator().getConfig().empty() ){
    LogInfo << "Generating post-fit sample plots..." << std::endl;
    getLikelihoodInterface().getPlotGenerator().generateSamplePlots(GenericToolbox::mkdirTFile(_saveDir_, "postFit/samples"));
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
      _parameterScanner_.scanSegment(GenericToolbox::mkdirTFile(_saveDir_, "postFit/converge"),
                                     _postFitParState_, _preFitParState_);
      GenericToolbox::triggerTFileWrite(_saveDir_);
    }
  }


  if( _minimizer_->getMinimizerStatus() != 0 ){
    LogError << "Skipping post-fit error calculation since the minimizer did not converge." << std::endl;
  }
  else{
    if( _minimizer_->isErrorCalcEnabled() ){
#ifdef GUNDAM_USING_CACHE_MANAGER
      // To calculate the llh, we only need to grab the bin content, not
      // individual weight
      bool origCopy = Cache::Manager::SetIsEventWeightCopyEnabled( false );
#endif

      LogInfo << "Computing post-fit errors..." << std::endl;
      _minimizer_->calcErrors();

#ifdef GUNDAM_USING_CACHE_MANAGER
      if( Cache::Manager::IsBuilt() ){
        LogInfo << "Copy Cache::Manager weights after computing post fit error"
                << std::endl;
        Cache::Manager::Get()->CopyEventWeights();
      }
      // return to default behavior
      Cache::Manager::SetIsEventWeightCopyEnabled( origCopy);
#endif
    }
  }

  LogInfo << "Fit is done." << std::endl;
}

// protected
void FitterEngine::runPcaCheck(){

  getLikelihoodInterface().propagateAndEvalLikelihood();

  LogAlert << "Using PCA method: " << _pcaMethod_.toString() << " / threshold = " << _pcaThreshold_ << std::endl;

  double baseLlh = getLikelihoodInterface().getLastLikelihood();
  double baseLlhStat = getLikelihoodInterface().getLastStatLikelihood();
  double baseLlhSyst = getLikelihoodInterface().getLastPenaltyLikelihood();

  LogInfo << "Reference stat log-likelihood for PCA: " << baseLlhStat << std::endl;

  // +1 sigma
  int iFitPar = -1;
  std::stringstream ssPrint;

  for( auto& parSet : getLikelihoodInterface().getModelPropagator().getParametersManager().getParameterSetsList() ){

    if( not parSet.isEnabled() ){ continue; }

    if( not parSet.isEnablePca() ){
      LogInfo << "PCA disabled on " << parSet.getName() << ". Skipping..." << std::endl;
      continue;
    }
    else{
      LogAlert << "Performing PCA on " << parSet.getName() << "..." << std::endl;
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
        LogAlert << red << ssPrint.str() << " -> FIXED" << rst << std::endl;
        continue;
      }

      if( par.isEnabled() and not par.isFixed() ){
        double currentParValue = par.getParameterValue();
        par.setParameterValue( currentParValue + par.getStdDevValue() );

        ssPrint << " " << currentParValue << " -> " << par.getParameterValue();
        LogInfo << ssPrint.str() << "..." << std::endl;

        getLikelihoodInterface().propagateAndEvalLikelihood();

        double criteria{0};

        bool fixParPca{false};
        if( _pcaMethod_ == PcaMethod::DeltaChi2Threshold ){
          criteria = std::abs(getLikelihoodInterface().getLastStatLikelihood() - baseLlhStat);
          ssPrint << ": deltaChi2Stat = " << criteria;
        }
        else if( _pcaMethod_ == PcaMethod::ReducedDeltaChi2Threshold ){
          criteria = std::abs(getLikelihoodInterface().getLastStatLikelihood() - baseLlhStat)/_minimizer_->fetchNbDegreeOfFreedom();
          ssPrint << ": deltaChi2Stat/dof = " << criteria;
        }
        else if( _pcaMethod_ == PcaMethod::SqrtReducedDeltaChi2Threshold ){
          criteria = std::sqrt( std::abs(getLikelihoodInterface().getLastStatLikelihood() - baseLlhStat)/_minimizer_->fetchNbDegreeOfFreedom() );
          ssPrint << ": sqrt( deltaChi2Stat/dof ) = " << criteria;
        }

        // color str
        std::string color;
        std::string rst;

        if( criteria < _pcaThreshold_ ){
          ssPrint << " < " << _pcaThreshold_ << " -> FIXED";
          fixParPca = true;
#ifndef NOCOLOR
          color = GenericToolbox::ColorCodes::yellowLightText;
          rst = GenericToolbox::ColorCodes::resetColor;
#endif
        }
        else{
          fixParPca = false;
          ssPrint << " >= " << _pcaThreshold_ << " -> OK";
        }

        LogInfo.moveTerminalCursorBack(1); // clear the old line
        LogAlert << color << ssPrint.str() << rst << std::endl;

        if( fixParPca ){
          par.setIsFixed(true); // ignored in the Chi2 computation of the parSet
          if( parSet.isEnableEigenDecomp() and _config_.fetchValue("fixGhostEigenParametersAfterFirstRejected", false) ){
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
  getLikelihoodInterface().propagateAndEvalLikelihood();
}
void FitterEngine::rescaleParametersStepSize(){
  LogInfo << __METHOD_NAME__ << std::endl;

  getLikelihoodInterface().propagateAndEvalLikelihood();
  double baseLlhPull = getLikelihoodInterface().getLastPenaltyLikelihood();
  double baseLlh = getLikelihoodInterface().getLastLikelihood();

  // +1 sigma
  for( auto& parSet : getLikelihoodInterface().getModelPropagator().getParametersManager().getParameterSetsList() ){

    for( auto& par : parSet.getEffectiveParameterList() ){

      if( not par.isEnabled() ){ continue; }

      double currentParValue = par.getParameterValue();
      par.setParameterValue( currentParValue + par.getStdDevValue() );

      getLikelihoodInterface().propagateAndEvalLikelihood();

      double deltaChi2 = getLikelihoodInterface().getLastLikelihood() - baseLlh;
      double deltaChi2Pulls = getLikelihoodInterface().getLastPenaltyLikelihood() - baseLlhPull;

      // Consider a parabolic approx:
      // only rescale with X2 stat?
//        double stepSize = std::sqrt(deltaChi2Pulls)/std::sqrt(deltaChi2);

      // full rescale
      double stepSize = 1./std::sqrt(std::abs(deltaChi2));

      LogInfo << "Step size of " << parSet.getName() + "/" + par.getTitle()
              << " -> σ x " << _parStepGain_ << " x " << stepSize
              << " -> Δχ² = " << deltaChi2 << " = " << deltaChi2 - deltaChi2Pulls << "(stat) + " << deltaChi2Pulls << "(pulls)";

      stepSize *= par.getStdDevValue() * _parStepGain_;

      par.setStepSize( stepSize );
      par.setParameterValue( currentParValue + stepSize );
      getLikelihoodInterface().propagateAndEvalLikelihood();
      LogInfo << " -> Δχ²(step) = " << getLikelihoodInterface().getLastLikelihood() - baseLlh << std::endl;
      par.setParameterValue( currentParValue );
    }

  }

  getLikelihoodInterface().propagateAndEvalLikelihood();
}
bool FitterEngine::checkNumericalAccuracy(){
  LogAlert << __METHOD_NAME__ << std::endl;
  int nTest{100}; int nThrows{10}; double gain{20};
  std::vector<std::vector<std::vector<double>>> throws(nThrows); // saved throws [throw][parSet][par]
  std::vector<double> responses(nThrows, std::nan("unset"));
  // stability/numerical accuracy test

  bool accurate = true;

  LogAlert << "Throwing..." << std::endl;
  for(auto& throwEntry : throws ){
    for( auto& parSet : getLikelihoodInterface().getModelPropagator().getParametersManager().getParameterSetsList() ){
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

  LogAlert << "Testing..." << std::endl;
  for( int iTest = 0 ; iTest < nTest ; iTest++ ){
    GenericToolbox::displayProgressBar(iTest, nTest, "Testing computational accuracy...");
    for( size_t iThrow = 0 ; iThrow < throws.size() ; iThrow++ ){
      int iParSet{-1};
      for( auto& parSet : getLikelihoodInterface().getModelPropagator().getParametersManager().getParameterSetsList() ){
        if(not parSet.isEnabled()) continue;
        if( not parSet.isEnabledThrowToyParameters() ){ continue;}
        iParSet++;
        for( size_t iPar = 0 ; iPar < parSet.getParameterList().size() ; iPar++){
          parSet.getParameterList()[iPar].setParameterValue( throws[iThrow][iParSet][iPar] );
        }
      }
      std::future<bool> eventually = getLikelihoodInterface().getModelPropagator().applyParameters();
      getLikelihoodInterface().evalLikelihood(eventually);

      if( responses[iThrow] == responses[iThrow] ){ // not nan
        if (not GundamUtils::almostEqual(
                getLikelihoodInterface().getLastLikelihood(),
                responses[iThrow], 0.02)) {
          accurate = false;
          // Check for 50 bits (out of 53 bits) of equality
          LogError << "Throw " << iThrow << " not accurate: "
                  << getLikelihoodInterface().getLastLikelihood() - responses[iThrow]
                  << " for " << GET_VAR_NAME_VALUE(getLikelihoodInterface().getLastLikelihood())
                  << " <=> " << GET_VAR_NAME_VALUE(responses[iThrow])
                  << std::endl;
        }
      }
      responses[iThrow] = getLikelihoodInterface().getLastLikelihood();
    }
    LogDebug << GenericToolbox::toString(responses) << std::endl;
  }
  if (accurate) LogAlert  << "Numeric accuracy test passed" << std::endl;

  return accurate;
}
