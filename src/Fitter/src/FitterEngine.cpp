//
// Created by Nadrino on 11/06/2021.
//

#include "FitterEngine.h"
#include "GundamGlobals.h"
#include "MinimizerInterface.h"
#include "MCMCInterface.h"

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

  _enablePca_ = GenericToolbox::Json::fetchValue(_config_, std::vector<std::string>{"enablePca", "fixGhostFitParameters"}, _enablePca_);
  _pcaDeltaChi2Threshold_ = GenericToolbox::Json::fetchValue(_config_, {{"ghostParameterDeltaChi2Threshold"}, {"pcaDeltaChi2Threshold"}}, _pcaDeltaChi2Threshold_);

  _enablePreFitScan_ = GenericToolbox::Json::fetchValue(_config_, "enablePreFitScan", _enablePreFitScan_);
  _enablePostFitScan_ = GenericToolbox::Json::fetchValue(_config_, "enablePostFitScan", _enablePostFitScan_);
  _enablePreFitToPostFitLineScan_ = GenericToolbox::Json::fetchValue(_config_, "enablePreFitToPostFitLineScan", _enablePreFitToPostFitLineScan_);

  _generateSamplePlots_ = GenericToolbox::Json::fetchValue(_config_, "generateSamplePlots", _generateSamplePlots_);
  _generateOneSigmaPlots_ = GenericToolbox::Json::fetchValue(_config_, "generateOneSigmaPlots", _generateOneSigmaPlots_);
  _doAllParamVariations_ = GenericToolbox::Json::doKeyExist(_config_, "allParamVariations");
  _allParamVariationsSigmas_ = GenericToolbox::Json::fetchValue(_config_, "allParamVariations", _allParamVariationsSigmas_);

  _scaleParStepWithChi2Response_ = GenericToolbox::Json::fetchValue(_config_, "scaleParStepWithChi2Response", _scaleParStepWithChi2Response_);
  _parStepGain_ = GenericToolbox::Json::fetchValue(_config_, "parStepGain", _parStepGain_);

  _throwMcBeforeFit_ = GenericToolbox::Json::fetchValue(_config_, "throwMcBeforeFit", _throwMcBeforeFit_);
  _throwGain_ = GenericToolbox::Json::fetchValue(_config_, "throwMcBeforeFitGain", _throwGain_);

  _propagator_.readConfig( GenericToolbox::Json::fetchValue<JsonType>(_config_, "propagatorConfig") );
  _savePostfitEventTrees_ = GenericToolbox::Json::fetchValue(_config_, "savePostfitEventTrees", _savePostfitEventTrees_);


  std::string engineType = GenericToolbox::Json::fetchValue(_config_,"engineType","minimizer");

  if (engineType == "minimizer") {
      this->_minimizer_ = std::make_unique<MinimizerInterface>(this);
      getMinimizer().readConfig( GenericToolbox::Json::fetchValue(_config_, "minimizerConfig", JsonType()));
  }
  else if (engineType == "mcmc") {
      this->_minimizer_ = std::make_unique<MCMCInterface>(this);
      getMinimizer().readConfig( GenericToolbox::Json::fetchValue(_config_, "mcmcConfig", JsonType()));
  }
  else {
      LogWarning << "Allowed engine types: minimizer, mcmc" << std::endl;
      LogThrow("Illegal engine type: \"" + engineType + "\"");
  }


  // legacy
  GenericToolbox::Json::deprecatedAction(_config_, "scanConfig", [&]{
    LogAlert << "Forwarding the option to Propagator. Consider moving it into \"propagatorConfig:\"" << std::endl;
    _propagator_.getParScanner().readConfig( GenericToolbox::Json::fetchValue(_config_, "scanConfig", JsonType()) );
  });

  GenericToolbox::Json::deprecatedAction(_config_, "monitorRefreshRateInMs", [&]{
    LogAlert << "Forwarding the option to Propagator. Consider moving it into \"minimizerConfig:\"" << std::endl;
    getLikelihood().getConvergenceMonitor().setMaxRefreshRateInMs(GenericToolbox::Json::fetchValue<int>(_config_, "monitorRefreshRateInMs"));
  });

  LogInfo << "Convergence monitor will be refreshed every " << _likelihood_.getConvergenceMonitor().getMaxRefreshRateInMs() << "ms." << std::endl;
}
void FitterEngine::initializeImpl(){
  LogThrowIf(_config_.empty(), "Config is not set.");
  LogThrowIf(_saveDir_== nullptr);

  if( GundamGlobals::isLightOutputMode() ){
    LogWarning << "Light mode enabled, wiping plot gen config..." << std::endl;
    _propagator_.getPlotGenerator().readConfig(JsonType());
  }

  _propagator_.initialize();

  if( _propagator_.isThrowAsimovToyParameters() ){
    LogInfo << "Writing throws in TTree..." << std::endl;
    auto* throwsTree = new TTree("throws", "throws");

    std::vector<GenericToolbox::RawDataArray> thrownParameterValues{};
    thrownParameterValues.reserve(_propagator_.getParametersManager().getParameterSetsList().size());
    for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){
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
    this->fixGhostFitParameters();
  }

  // This moves the parameters
  if( _scaleParStepWithChi2Response_ ){
    LogInfo << "Using parameter step scale: " << _parStepGain_ << std::endl;
    this->rescaleParametersStepSize();
  }

  // The likelihood needs everything to be fully setup before it is initialized.
  getLikelihood().initialize();

  // The minimizer needs all the parameters to be fully setup (i.e. PCA done
  // and other properties)
  getMinimizer().initialize();

  if( GundamGlobals::getVerboseLevel() >= VerboseLevel::MORE_PRINTOUT ){ checkNumericalAccuracy(); }

  // Write data
  LogInfo << "Writing propagator objects..." << std::endl;
  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile(_saveDir_, "propagator"),
      TNamed("initialParameterState", GenericToolbox::Json::toReadableString(_propagator_.getParametersManager().exportParameterInjectorConfig()).c_str())
  );

  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile(_saveDir_, "propagator"),
      _propagator_.getParametersManager().getGlobalCovarianceMatrix().get(), "globalCovarianceMatrix"
  );
  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile(_saveDir_, "propagator"),
      _propagator_.getParametersManager().getStrippedCovarianceMatrix().get(), "strippedCovarianceMatrix"
  );
  for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){
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

  if( dynamic_cast<const MinimizerInterface*>( &this->getMinimizer() ) ){
    dynamic_cast<const MinimizerInterface*>( &this->getMinimizer() )->saveMinimizerSettings( GenericToolbox::mkdirTFile(_saveDir_, "fit/minimizer" ) );
  }

  this->_propagator_.updateLlhCache();

  if( not GundamGlobals::isLightOutputMode() ){
    _propagator_.getTreeWriter().writeSamples(GenericToolbox::mkdirTFile(_saveDir_, "preFit/events"));
  }

  // writing event rates
  LogInfo << "Writing event rates..." << std::endl;
  for( auto& sample : _propagator_.getSampleSet().getSampleList() ){
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
  LogThrowIf(not isInitialized());

  LogWarning << "Pre-fit likelihood state:" << std::endl;

  std::string llhState{_propagator_.getLlhBufferSummary()};
  LogInfo << llhState << std::endl;
  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile( _saveDir_, "preFit" ),
      TNamed("llhState", llhState.c_str())
  );
  _preFitParState_ = _propagator_.getParametersManager().exportParameterInjectorConfig();
  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile( _saveDir_, "preFit" ),
      TNamed("parState", GenericToolbox::Json::toReadableString(_preFitParState_).c_str())
  );

  // Not moving parameters
  if( _generateSamplePlots_ and not _propagator_.getPlotGenerator().getConfig().empty() ){
    LogInfo << "Generating pre-fit sample plots..." << std::endl;
    _propagator_.getPlotGenerator().generateSamplePlots(GenericToolbox::mkdirTFile(_saveDir_, "preFit/samples"));
    GenericToolbox::triggerTFileWrite(_saveDir_);
  }

  // Moving parameters
  if( _generateOneSigmaPlots_ and not _propagator_.getPlotGenerator().getConfig().empty() ){
    LogInfo << "Generating pre-fit one-sigma variation plots..." << std::endl;
    _propagator_.getParScanner().generateOneSigmaPlots(GenericToolbox::mkdirTFile(_saveDir_, "preFit/oneSigma"));
    GenericToolbox::triggerTFileWrite(_saveDir_);
  }
  if( _doAllParamVariations_ ){
    LogInfo << "Running all parameter variation on pre-fit samples..." << std::endl;
    _propagator_.getParScanner().varyEvenRates( _allParamVariationsSigmas_, GenericToolbox::mkdirTFile(_saveDir_, "preFit/varyEventRates") );
    GenericToolbox::triggerTFileWrite(_saveDir_);
  }
  if( _enablePreFitScan_ ){
    LogInfo << "Scanning fit parameters before minimizing..." << std::endl;
    getMinimizer().scanParameters(GenericToolbox::mkdirTFile(_saveDir_, "preFit/scan"));
    GenericToolbox::triggerTFileWrite(_saveDir_);
  }
  if( _throwMcBeforeFit_ ){
    LogAlert << "Throwing correlated parameters of MC away from their prior..." << std::endl;
    LogAlert << "Throw gain form MC push set to: " << _throwGain_ << std::endl;
    for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){
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
    this->_propagator_.updateLlhCache();
    LogAlert << _propagator_.getLlhBufferSummary() << std::endl;
  }

  // Leaving now?
  if( _isDryRun_ ){
    LogAlert << "Dry run requested. Leaving before the minimization." << std::endl;
    return;
  }

  LogInfo << "Minimizing LLH..." << std::endl;
  this->getMinimizer().minimize();

  LogWarning << "Saving post-fit par state..." << std::endl;
  _postFitParState_ = _propagator_.getParametersManager().exportParameterInjectorConfig();
  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile( _saveDir_, "postFit" ),
      TNamed("parState", GenericToolbox::Json::toReadableString(_postFitParState_).c_str())
  );

  LogWarning << "Post-fit likelihood state:" << std::endl;
  llhState = _propagator_.getLlhBufferSummary();
  LogInfo << llhState << std::endl;
  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile( _saveDir_, "postFit" ),
      TNamed("llhState", llhState.c_str())
  );


  if (_savePostfitEventTrees_){
      LogInfo << "Saving PostFit event Trees" << std::endl;
      _propagator_.getTreeWriter().writeSamples(GenericToolbox::mkdirTFile(_saveDir_, "postFit/events"));
  }
  if( _generateSamplePlots_ and not _propagator_.getPlotGenerator().getConfig().empty() ){
    LogInfo << "Generating post-fit sample plots..." << std::endl;
    _propagator_.getPlotGenerator().generateSamplePlots(GenericToolbox::mkdirTFile(_saveDir_, "postFit/samples"));
    GenericToolbox::triggerTFileWrite(_saveDir_);
  }
  if( _enablePostFitScan_ ){
    LogInfo << "Scanning fit parameters around the minimum point..." << std::endl;
    getMinimizer().scanParameters(GenericToolbox::mkdirTFile(_saveDir_, "postFit/scan"));
    GenericToolbox::triggerTFileWrite(_saveDir_);
  }
  if( _enablePreFitToPostFitLineScan_ ){
    if( not GundamGlobals::isLightOutputMode() ){
      LogInfo << "Scanning along the line from pre-fit to post-fit points..." << std::endl;
      getPropagator().getParScanner().scanSegment(GenericToolbox::mkdirTFile(_saveDir_, "postFit/scanConvergence"),
                                                  _postFitParState_, _preFitParState_);
      GenericToolbox::triggerTFileWrite(_saveDir_);
    }
  }

  if( getMinimizer().isFitHasConverged() and getMinimizer().isEnablePostFitErrorEval() ){
    LogInfo << "Computing post-fit errors..." << std::endl;
    getMinimizer().calcErrors();
  }
  else{
    if( not getMinimizer().isFitHasConverged() ) {
      LogError << "Skipping post-fit error calculation since the minimizer did not converge." << std::endl;
    }
    else{
      LogAlert << "Skipping post-fit error calculation since the option is disabled." << std::endl;
    }
  }

  LogWarning << "Fit is done." << std::endl;
}

// protected
void FitterEngine::fixGhostFitParameters(){

  _propagator_.updateLlhCache();
  double baseChi2 = _propagator_.getLlhBuffer();
  double baseChi2Stat = _propagator_.getLlhStatBuffer();
  double baseChi2Syst = _propagator_.getLlhPenaltyBuffer();

  LogInfo << "Reference " << GUNDAM_CHI2 << "(stat) for PCA: " << baseChi2Stat << std::endl;

  // +1 sigma
  int iFitPar = -1;
  std::stringstream ssPrint;
  double deltaChi2Stat;

  for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){

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
      ssPrint << "(" << par.getParameterIndex()+1 << "/" << parList.size() << ") +1" << GUNDAM_SIGMA << " on " << parSet.getName() + "/" + par.getTitle();

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

        _propagator_.updateLlhCache();
        deltaChi2Stat = _propagator_.getLlhStatBuffer() - baseChi2Stat;

        ssPrint << ": " << GUNDAM_DELTA << GUNDAM_CHI2 << " (stat) = " << deltaChi2Stat;

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
  _propagator_.updateLlhCache();
}

void FitterEngine::rescaleParametersStepSize(){
  LogInfo << __METHOD_NAME__ << std::endl;

  _propagator_.updateLlhCache();
  double baseChi2Pull = _propagator_.getLlhPenaltyBuffer();
  double baseChi2 = _propagator_.getLlhBuffer();

  // +1 sigma
  for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){

    for( auto& par : parSet.getEffectiveParameterList() ){

      if( not par.isEnabled() ){ continue; }

      double currentParValue = par.getParameterValue();
      par.setParameterValue( currentParValue + par.getStdDevValue() );

      _propagator_.updateLlhCache();

      double deltaChi2 = _propagator_.getLlhBuffer() - baseChi2;
      double deltaChi2Pulls = _propagator_.getLlhPenaltyBuffer() - baseChi2Pull;

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
      _propagator_.updateLlhCache();
      LogInfo << " -> Δχ²(step) = " << _propagator_.getLlhBuffer() - baseChi2 << std::endl;
      par.setParameterValue( currentParValue );
    }

  }

  _propagator_.updateLlhCache();
}
void FitterEngine::scanMinimizerParameters(TDirectory* saveDir_){
  LogThrowIf(not isInitialized());
  LogInfo << "Performing scans of fit parameters..." << std::endl;
  getMinimizer().scanParameters(saveDir_);
}
void FitterEngine::checkNumericalAccuracy(){
  LogWarning << __METHOD_NAME__ << std::endl;
  int nTest{100}; int nThrows{10}; double gain{20};
  std::vector<std::vector<std::vector<double>>> throws(nThrows); // saved throws [throw][parSet][par]
  std::vector<double> responses(nThrows, std::nan("unset"));
  // stability/numerical accuracy test

  LogInfo << "Throwing..." << std::endl;
  for(auto& throwEntry : throws ){
    for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){
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
      for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){
        if(not parSet.isEnabled()) continue;
        if( not parSet.isEnabledThrowToyParameters() ){ continue;}
        iParSet++;
        for( size_t iPar = 0 ; iPar < parSet.getParameterList().size() ; iPar++){
          parSet.getParameterList()[iPar].setParameterValue( throws[iThrow][iParSet][iPar] );
        }
      }
      _propagator_.updateLlhCache();

      if( responses[iThrow] == responses[iThrow] ){ // not nan
        LogThrowIf( _propagator_.getLlhBuffer() != responses[iThrow], "Not accurate: " << _propagator_.getLlhBuffer() - responses[iThrow] << " / "
                                                                        << GET_VAR_NAME_VALUE(_propagator_.getLlhBuffer()) << " <=> " << GET_VAR_NAME_VALUE(responses[iThrow])
        )
      }
      responses[iThrow] = _propagator_.getLlhBuffer();
    }
    LogDebug << GenericToolbox::toString(responses) << std::endl;
  }
  LogInfo << "OK" << std::endl;
}
