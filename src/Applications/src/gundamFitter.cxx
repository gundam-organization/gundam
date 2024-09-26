//
// Created by Nadrino on 01/06/2021.
//

#include "gundamFitter.hxx"

#include "GundamGlobals.h"
#include "FitterEngine.h"
#include "ConfigUtils.h"
#include "GundamUtils.h"
#include "GundamApp.h"
#ifdef GUNDAM_USING_CACHE_MANAGER
#include "CacheManager.h"
#endif

#include "GenericToolbox.Root.h"

#include "GenericToolbox.Map.h"
#include "CmdLineParser.h"
#include "Logger.h"

#include <string>
#include <vector>


int main(int argc, char** argv){
  
#ifdef GUNDAM_USING_CACHE_MANAGER
  if (Cache::Manager::HasCUDA()){ LogInfo << "CUDA inside" << std::endl; }
#endif

  GundamFitterApp myApp("main fitter", argc, argv);
  myApp.configure();
  myApp.initialize();
  myApp.run();

  GundamApp app{"main fitter"};

  // --------------------------
  // Init command line args:
  // --------------------------

  if( _clp_.isOptionTriggered("debugVerbose") ){ GundamGlobals::setIsDebugConfig( true ); }

  // Is build compatible with GPU option?
  if( _clp_.isOptionTriggered("usingGpu") ){
#ifdef GUNDAM_USING_CACHE_MANAGER
    LogThrowIf( not Cache::Manager::HasCUDA(), "CUDA support not enabled with this GUNDAM build." );
#else
    LogThrow("CUDA support not enabled with this GUNDAM build (GUNDAM_USING_CACHE_MANAGER required).")
#endif
    LogWarning << "Using GPU parallelization." << std::endl;
  }

  if (_clp_.isOptionTriggered("forceDirect")) GundamGlobals::setForceDirectCalculation(true);

  bool useCache = false;
#ifdef GUNDAM_USING_CACHE_MANAGER
  useCache = Cache::Manager::HasGPU(true);
#endif
  if (_clp_.isOptionTriggered("usingCacheManager")) {
      int values = _clp_.getNbValueSet("usingCacheManager");
      if (values < 1) useCache = not useCache;
      else if ("on" == _clp_.getOptionVal<std::string>("usingCacheManager",0)) useCache = true;
      else if ("off" == _clp_.getOptionVal<std::string>("usingCacheManager",0)) useCache = false;
      else {
          LogThrow("Invalid --cache-manager argument: must be empty, 'on' or 'off'");
      }
  }
  if (_clp_.isOptionTriggered("usingGpu")) useCache = true;

#ifdef GUNDAM_USING_CACHE_MANAGER
    GundamGlobals::setEnableCacheManager(useCache);
    if (not useCache) {
      LogWarning << "Cache::Manager enabled but turned off for job"
                 << std::endl;
    }
#else
    LogThrowIf(useCache, "GUNDAM compiled without Cache::Manager");
#endif

  // No cache on dials?
  if( _clp_.isOptionTriggered("noDialCache") ){
    LogAlert << "Disabling cache in dial evaluation (when available)..." << std::endl;
    GundamGlobals::setDisableDialCache(true);
  }

  // inject parameter config?
  std::string injectParameterPath{};
  if( _clp_.isOptionTriggered("injectParameterConfig") ){
    LogWarning << "Inject parameter config: " << _clp_.getOptionVal<std::string>("injectParameterConfig") << std::endl;
    injectParameterPath = _clp_.getOptionVal<std::string>("injectParameterConfig");
  }

  // toy par injector
  std::string toyParInjector{};
  if( _clp_.isOptionTriggered("injectToyParameters") ){
    toyParInjector = _clp_.getOptionVal<std::string>("injectToyParameters");
    LogWarning << "Inject toy parameter: " << toyParInjector << std::endl;
  }

  // PRNG seed?
  gRandom = new TRandom3(0);    // Initialize with a UUID;
  if( _clp_.isOptionTriggered("randomSeed") ){
    LogAlert << "Using user-specified random seed: " << _clp_.getOptionVal<ULong_t>("randomSeed") << std::endl;
    gRandom->SetSeed(_clp_.getOptionVal<ULong_t>("randomSeed"));
  }

  // How many parallel threads?
  GundamGlobals::setNumberOfThreads( _clp_.getOptionVal("nbThreads", 1) );
  LogInfo << "Running the fitter with " << GundamGlobals::getNumberOfThreads() << " parallel threads." << std::endl;

  // Reading configuration
  auto configFilePath = _clp_.getOptionVal("configFile", "");
  LogThrowIf(configFilePath.empty(), "Config file not provided.");

  ConfigUtils::ConfigHandler configHandler(configFilePath);
  configHandler.override( _clp_.getOptionValList<std::string>("overrideFiles") );
  configHandler.flatOverride( _clp_.getOptionValList<std::string>("overrides") );

  auto gundamFitterConfig(configHandler.getConfig());

  // Output file path
  std::string outFileName;
  if( _clp_.isOptionTriggered("outputFilePath") ){
    outFileName = _clp_.getOptionVal("outputFilePath", outFileName + ".root");
  }
  else{

    std::string outFolder{"./"};
    GenericToolbox::Json::fillValue(gundamFitterConfig, outFolder, "outputFolder");
    if( _clp_.isOptionTriggered("outputDir") ){ outFolder = _clp_.getOptionVal<std::string>("outputDir"); }

    // appendixDict["optionName"] = "Appendix"
    // this list insure all appendices will appear in the same order
    std::vector<GundamUtils::AppendixEntry> appendixDict{
        {"configFile", ""},
        {"overrideFiles", "With"},
        {"injectParameterConfig", "Inj"},
        {"scanLine", "LineSc"},
        {"useDataEntry", "DataEntry"},
        {"asimov", "Asimov"},
        {"scanParameters", "Scan"},
        {"generateOneSigmaPlots", "OneSigma"},
        {"enablePca", "PCA"},
        {"skipHesse", "NoHesse"},
        {"kickMc", "KickMc"},
        {"lightOutputMode", "Light"},
        {"toyFit", "ToyFit"},
        {"injectToyParameters", "InjToyPar"},

        // debug options
        {"debugMaxNbEventToLoad", "debugNbEventMax"},

        // trailing
        {"dry-run", "DryRun"},
        {"appendix", ""},
    };

    outFileName = GenericToolbox::joinPath(
        outFolder,
        "gundamFitter_" + GundamUtils::generateFileName(_clp_, appendixDict)
    ) + ".root";
  }


  // --------------------------
  // Initialize the fitter:
  // --------------------------

  // Checking the minimal version for the config
  if( _clp_.isOptionTriggered("ignoreVersionCheck") ){
    LogAlert << "Ignoring GUNDAM version check." << std::endl;
  }
  else{
    std::string minGundamVersion("0.0.0");
    GenericToolbox::Json::fillValue(gundamFitterConfig, minGundamVersion, "minGundamVersion");
    LogThrowIf(
        not GundamUtils::isNewerOrEqualVersion( minGundamVersion ),
        "Version check FAILED: " << GundamUtils::getVersionStr() << " < " << minGundamVersion
    );
    LogInfo << "Version check passed: ";
    LogInfo << GundamUtils::getVersionStr() << " >= " << minGundamVersion << std::endl;
  }

  // to write cmdLine info
  app.setCmdLinePtr( &_clp_ );

  // unfolded config
  app.setConfigString( GenericToolbox::Json::toReadableString(gundamFitterConfig) );

  // Ok, we should run. Create the out file.
  app.openOutputFile(outFileName);
  app.writeAppInfo();


  // --------------------------
  // Configure:
  // --------------------------
  LogInfo << "FitterEngine setup..." << std::endl;
  FitterEngine fitter(GenericToolbox::mkdirTFile(app.getOutfilePtr(), "FitterEngine"));

  GenericToolbox::Json::fillValue(gundamFitterConfig, fitter.getConfig(), "fitterEngineConfig");
  fitter.configure();

  // -a
  if( _clp_.isOptionTriggered("asimov") ){
    fitter.getLikelihoodInterface().setForceAsimovData( true );
    fitter.getLikelihoodInterface().setDataType( LikelihoodInterface::DataType::Asimov );
  }
  else{
    // by default, assume it's a real/fake data fit
    fitter.getLikelihoodInterface().setDataType( LikelihoodInterface::DataType::RealData );
  }


  // --use-data-entry
  if( _clp_.isOptionTriggered("useDataEntry") ){
    auto selectedDataEntry = _clp_.getOptionVal<std::string>("useDataEntry", 0);
    // Do something better in case multiple datasets are defined
    bool isFound{false};
    for( auto& dataSet : fitter.getLikelihoodInterface().getDatasetList() ){
      if( GenericToolbox::isIn( selectedDataEntry, dataSet.getDataDispenserDict() ) ){
        LogWarning << "Using data entry \"" << selectedDataEntry << "\" for dataset: " << dataSet.getName() << std::endl;
        dataSet.setSelectedDataEntry( selectedDataEntry );
        isFound = true;
      }
    }
    LogThrowIf(not isFound, "Could not find data entry \"" << selectedDataEntry << "\" among defined data sets");
  }

  // --skip-hesse
  fitter.getMinimizer().setDisableCalcError( _clp_.isOptionTriggered("skipHesse") );

  // --scan <N>
  if( _clp_.isOptionTriggered("scanParameters") ) {
    fitter.setEnablePreFitScan( true );
    fitter.setEnablePostFitScan( true );
    fitter.getParameterScanner().setNbPoints(
        _clp_.getOptionVal("scanParameters", fitter.getParameterScanner().getNbPoints())
    );
  }

  // --enable-pca
  if( _clp_.isOptionTriggered("enablePca") ){
    fitter.setEnablePca( true );
    if( _clp_.getNbValueSet("enablePca") == 1 ){
      fitter.setPcaThreshold( _clp_.getOptionVal<double>("enablePca") );
    }
    if( _clp_.getNbValueSet("enablePca") == 2 ){
      fitter.setPcaThreshold( _clp_.getOptionVal<double>("enablePca", 0) );
      fitter.setPcaMethod( FitterEngine::PcaMethod::toEnum(_clp_.getOptionVal<std::string>("enablePca", 1), true) );
    }
  }

  // --toy <iToy>
  if( _clp_.isOptionTriggered("toyFit") ){
    fitter.getLikelihoodInterface().setDataType( LikelihoodInterface::DataType::Toy );
    fitter.getLikelihoodInterface().getModelPropagator().setIThrow( _clp_.getOptionVal("toyFit", -1) );
  }

  // -d
  fitter.setIsDryRun( _clp_.isOptionTriggered("dry-run") );

  // --one-sigma
  fitter.setGenerateOneSigmaPlots( _clp_.isOptionTriggered("generateOneSigmaPlots") );

  // --light-mode
  GundamGlobals::setLightOutputMode(_clp_.isOptionTriggered("lightOutputMode") );

  // injectParameterPath
  if( not injectParameterPath.empty() ){
    auto injectConfig = ConfigUtils::readConfigFile( injectParameterPath );
    fitter.getLikelihoodInterface().getModelPropagator().setParameterInjectorConfig(injectConfig);
  }

  // toyParInjector
  if( not toyParInjector.empty() ){
    auto injectConfig = ConfigUtils::readConfigFile( toyParInjector );
    fitter.getLikelihoodInterface().setToyParameterInjector( injectConfig );
  }

  // Also check app level config options
  GenericToolbox::Json::deprecatedAction(gundamFitterConfig, "generateSamplePlots", [&]{
    LogAlert << "Forwarding the option to FitterEngine. Consider moving it into \"fitterEngineConfig:\"" << std::endl;
    fitter.setGenerateSamplePlots( GenericToolbox::Json::fetchValue<bool>(gundamFitterConfig, "generateSamplePlots") );
  });

  GenericToolbox::Json::deprecatedAction(gundamFitterConfig, "allParamVariations", [&]{
    LogAlert << "Forwarding the option to FitterEngine. Consider moving it into \"fitterEngineConfig:\"" << std::endl;
    fitter.setDoAllParamVariations(true);
    fitter.setAllParamVariationsSigmas(GenericToolbox::Json::fetchValue<std::vector<double>>(gundamFitterConfig, "allParamVariations"));
  });

  // Check if the first point of the fit should be moved before the
  // minimization.  This is not changing the prior value, only the starting
  // point of the fit.  The kick is in units of prior standard deviations
  // about the prior point.
  double kickMc = 0.0;          // Set the default value without an option
  if (_clp_.isOptionTriggered("kickMc")) {
      int values = _clp_.getNbValueSet("usingCacheManager");
      if (values < 1) kickMc = 0.1; // Default without an argument.
      else kickMc = _clp_.getOptionVal<double>("kickMc",0);
  }
  if ( kickMc > 0.01 ) {
      LogAlert << "Fit starting point randomized by " << kickMc << " sigma"
               << " around prior values."
               << std::endl;
      fitter.setThrowMcBeforeFit( true );
      fitter.setThrowGain( kickMc );
  }

  if( _clp_.isOptionTriggered("debugMaxNbEventToLoad") ){
    LogThrowIf(_clp_.getNbValueSet("debugMaxNbEventToLoad") != 1, "Nb of event not specified.");
    LogDebug << "Load " << _clp_.getOptionVal<size_t>("debugMaxNbEventToLoad") << "max events per dataset." << std::endl;
    for( auto& dataset : fitter.getLikelihoodInterface().getDatasetList() ){
      dataset.setNbMaxEventToLoad(_clp_.getOptionVal<size_t>("debugMaxNbEventToLoad"));
    }
  }

  // --------------------------
  // Load:
  // --------------------------
  fitter.initialize();

  // show initial conditions
  if( _clp_.isOptionTriggered("injectParameterConfig") ) {
    LogDebug << "Starting mc parameters that where injected:" << std::endl;
    LogDebug << fitter.getLikelihoodInterface().getModelPropagator().getParametersManager().getParametersSummary(false ) << std::endl;
  }

  if( _clp_.isOptionTriggered("scanLine") ){
    auto* outDir = GenericToolbox::mkdirTFile(fitter.getSaveDir(), GenericToolbox::joinPath("preFit", "cmdScanLine"));
    LogThrowIf( _clp_.getNbValueSet("scanLine") == 0, "No injector file provided.");
    if( _clp_.getNbValueSet("scanLine") == 1 ){
      LogAlert << "Will scan the line toward the point set in: " << _clp_.getOptionVal<std::string>("scanLine", 0) << std::endl;

      auto endPoint = ConfigUtils::readConfigFile(_clp_.getOptionVal<std::string>("scanLine", 0));

      GenericToolbox::writeInTFile( outDir, TNamed("endPoint", GenericToolbox::Json::toReadableString(endPoint).c_str()) );

      fitter.getParameterScanner().scanSegment( outDir, endPoint );
    }
    else if( _clp_.getNbValueSet("scanLine") == 2 ) {
      LogAlert << "Will scan the line from point A (" << _clp_.getOptionVal<std::string>("scanLine", 0)
          << ") to point B (" << _clp_.getOptionVal<std::string>("scanLine", 1) << ")" << std::endl;

      auto startPoint = ConfigUtils::readConfigFile(_clp_.getOptionVal<std::string>("scanLine", 0));
      auto endPoint = ConfigUtils::readConfigFile(_clp_.getOptionVal<std::string>("scanLine", 1));

      GenericToolbox::writeInTFile( outDir, TNamed("startPoint", GenericToolbox::Json::toReadableString(startPoint).c_str()) );
      GenericToolbox::writeInTFile( outDir, TNamed("endPoint", GenericToolbox::Json::toReadableString(endPoint).c_str()) );

      fitter.getParameterScanner().scanSegment( outDir, endPoint, startPoint );
    }
    else{
      LogThrow("");
    }
  }

  // --------------------------
  // Run the fitter:
  // --------------------------
  fitter.fit();

}

void GundamFitterApp::defineCommandLineOptions(){
  CmdLineParserGlobals::_fascistMode_ = true;

  _clp_.getDescription() << "> gundamFitter is the main interface for the fitter." << std::endl;
  _clp_.getDescription() << "> " << std::endl;
  _clp_.getDescription() << "> It takes a set of inputs through config files and command line argument," << std::endl;
  _clp_.getDescription() << "> and initialize the fitter engine." << std::endl;
  _clp_.getDescription() << "> Once ready, the fitter minimize the likelihood function and" << std::endl;
  _clp_.getDescription() << "> produce a set of plot saved in the output ROOT file." << std::endl;

  _clp_.addDummyOption("Main options");

  _clp_.addOption("configFile", {"-c", "--config-file"}, "Specify path to the fitter config file");
  _clp_.addOption("nbThreads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");
  _clp_.addOption("outputFilePath", {"-o", "--out-file"}, "Specify the output file");
  _clp_.addOption("outputDir", {"--out-dir"}, "Specify the output directory");
  _clp_.addOption("randomSeed", {"-s", "--seed"}, "Set random seed");
  _clp_.addOption("useDataEntry", {"--use-data-entry"}, "Overrides \"selectedDataEntry\" in dataSet config. Second arg is to select a given dataset");
  _clp_.addOption("injectParameterConfig", {"--inject-parameters"}, "Inject parameters defined in the provided config file");
  _clp_.addOption("injectToyParameters", {"--inject-toy-parameter"}, "Inject parameters defined in the provided config file");
  _clp_.addOption("appendix", {"--appendix"}, "Add appendix to the output file name");

  _clp_.addDummyOption("Trigger options");

  _clp_.addTriggerOption("dry-run", {"--dry-run", "-d"},"Perform the full sequence of initialization, but don't do the actual fit.");
  _clp_.addTriggerOption("asimov", {"-a", "--asimov"}, "Use MC dataset to fill the data histograms");
  _clp_.addTriggerOption("skipHesse", {"--skip-hesse"}, "Don't perform postfit error evaluation");
  _clp_.addTriggerOption("generateOneSigmaPlots", {"--one-sigma"}, "Generate one sigma plots");
  _clp_.addTriggerOption("lightOutputMode", {"--light-mode"}, "Disable plot generation");
  _clp_.addTriggerOption("noDialCache", {"--no-dial-cache"}, "Disable cache handling for dial eval");
  _clp_.addTriggerOption("ignoreVersionCheck", {"--ignore-version"}, "Don't check GUNDAM version with config request");

  _clp_.addOption("scanParameters", {"--scan"}, "Enable parameter scan before and after the fit (can provide nSteps)", 1, true);
  _clp_.addOption("scanLine", {"--scan-line"}, "Provide par injector files: start and end point or only end point (start will be prefit)", 2, true);
  _clp_.addOption("toyFit", {"--toy"}, "Run a toy fit (optional arg to provide toy index)", 1, true);
  _clp_.addOption("enablePca", {"--pca", "--enable-pca"}, "Enable principle component analysis for eigen decomposed parameter sets", 2, true);

  _clp_.addDummyOption("Runtime options");

  _clp_.addOption("kickMc", {"--kick-mc"}, "Amount to push the starting parameters away from their prior values (default: 0)", 1, true);
  _clp_.addOption("debugVerbose", {"--debug"}, "Enable debug verbose (can provide verbose level arg)", 1, true);
  _clp_.addOption("usingCacheManager", {"--cache-manager"}, "Toggle the usage of the CacheManager (i.e. the GPU) [empty, 'on', or 'off']",1,true);
  _clp_.addTriggerOption("usingGpu", {"--gpu"}, "Use GPU parallelization");
  _clp_.addOption("overrides", {"-O", "--override"}, "Add a config override [e.g. /fitterEngineConfig/engineType=mcmc)", -1);
  _clp_.addOption("overrideFiles", {"-of", "--override-files"}, "Provide config files that will override keys", -1);

  _clp_.addDummyOption("Debugging options");
  _clp_.addTriggerOption("forceDirect", {"--cpu"}, "Force direct calculation of weights (for debugging)");
  _clp_.addOption("debugMaxNbEventToLoad", {"-me", "--max-events"}, "Set the maximum number of events to load per dataset", 1);

  _clp_.addDummyOption();

  this->GundamAppTemplate::defineCommandLineOptions();
}


void GundamFitterApp::configureImpl(){
  this->defineCommandLineOptions();
  this->readCommandLineOptions();
}
