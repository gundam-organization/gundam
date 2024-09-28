//
// Created by Nadrino on 01/06/2021.
//

#include "GundamGlobals.h"
#include "FitterEngine.h"
#include "RootMinimizer.h"
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

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::getUserHeader() << "[" << FILENAME << "]"; });
#endif


int main(int argc, char** argv){

  GundamApp app{"main fitter"};

#ifdef GUNDAM_USING_CACHE_MANAGER
  if (Cache::Manager::HasCUDA()){ LogInfo << "CUDA inside" << std::endl; }
#endif

  // --------------------------
  // Read Command Line Args:
  // --------------------------
  CmdLineParser clParser;

  CmdLineParserGlobals::_fascistMode_ = true;

  clParser.getDescription() << "> gundamFitter is the main interface for the fitter." << std::endl;
  clParser.getDescription() << "> " << std::endl;
  clParser.getDescription() << "> It takes a set of inputs through config files and command line argument," << std::endl;
  clParser.getDescription() << "> and initialize the fitter engine." << std::endl;
  clParser.getDescription() << "> Once ready, the fitter minimize the likelihood function and" << std::endl;
  clParser.getDescription() << "> produce a set of plot saved in the output ROOT file." << std::endl;

  clParser.addDummyOption("Main options");

  clParser.addOption("configFile", {"-c", "--config-file"}, "Specify path to the fitter config file");
  clParser.addOption("nbThreads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");
  clParser.addOption("outputFilePath", {"-o", "--out-file"}, "Specify the output file");
  clParser.addOption("outputDir", {"--out-dir"}, "Specify the output directory");
  clParser.addOption("randomSeed", {"-s", "--seed"}, "Set random seed");
  clParser.addOption("useDataEntry", {"--use-data-entry"}, "Overrides \"selectedDataEntry\" in dataSet config. Second arg is to select a given dataset");
  clParser.addOption("injectParameterConfig", {"--inject-parameters"}, "Inject parameters defined in the provided config file");
  clParser.addOption("injectToyParameters", {"--inject-toy-parameter"}, "Inject parameters defined in the provided config file");
  clParser.addOption("appendix", {"--appendix"}, "Add appendix to the output file name");

  clParser.addDummyOption("Trigger options");

  clParser.addTriggerOption("dry-run", {"--dry-run", "-d"},"Perform the full sequence of initialization, but don't do the actual fit.");
  clParser.addTriggerOption("asimov", {"-a", "--asimov"}, "Use MC dataset to fill the data histograms");
  clParser.addTriggerOption("skipHesse", {"--skip-hesse"}, "Don't perform postfit error evaluation");
  clParser.addTriggerOption("generateOneSigmaPlots", {"--one-sigma"}, "Generate one sigma plots");
  clParser.addTriggerOption("lightOutputMode", {"--light-mode"}, "Disable plot generation");
  clParser.addTriggerOption("noDialCache", {"--no-dial-cache"}, "Disable cache handling for dial eval");
  clParser.addTriggerOption("ignoreVersionCheck", {"--ignore-version"}, "Don't check GUNDAM version with config request");

  clParser.addOption("scanParameters", {"--scan"}, "Enable parameter scan before and after the fit (can provide nSteps)", 1, true);
  clParser.addOption("scanLine", {"--scan-line"}, "Provide par injector files: start and end point or only end point (start will be prefit)", 2, true);
  clParser.addOption("toyFit", {"--toy"}, "Run a toy fit (optional arg to provide toy index)", 1, true);
  clParser.addOption("enablePca", {"--pca", "--enable-pca"}, "Enable principle component analysis for eigen decomposed parameter sets", 2, true);

  clParser.addDummyOption("Runtime options");

  clParser.addOption("kickMc", {"--kick-mc"}, "Amount to push the starting parameters away from their prior values (default: 0)", 1, true);
  clParser.addOption("debugVerbose", {"--debug"}, "Enable debug verbose (can provide verbose level arg)", 1, true);
  clParser.addOption("usingCacheManager", {"--cache-manager"}, "Toggle the usage of the CacheManager (i.e. the GPU) [empty, 'on', or 'off']",1,true);
  clParser.addTriggerOption("usingGpu", {"--gpu"}, "Use GPU parallelization");
  clParser.addOption("overrides", {"-O", "--override"}, "Add a config override [e.g. /fitterEngineConfig/engineType=mcmc)", -1);
  clParser.addOption("overrideFiles", {"-of", "--override-files"}, "Provide config files that will override keys", -1);

  clParser.addDummyOption("Debugging options");
  clParser.addTriggerOption("forceDirect", {"--cpu"}, "Force direct calculation of weights (for debugging)");
  clParser.addOption("debugMaxNbEventToLoad", {"-me", "--max-events"}, "Set the maximum number of events to load per dataset", 1);

  clParser.addDummyOption();


  LogInfo << clParser.getDescription().str() << std::endl;

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogThrowIf(clParser.isNoOptionTriggered(), "No option was provided.");

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;
  LogInfo << clParser.dumpConfigAsJsonStr() << std::endl;


  // --------------------------
  // Init command line args:
  // --------------------------

  if( clParser.isOptionTriggered("debugVerbose") ){ GundamGlobals::setIsDebugConfig( true ); }

  // Is build compatible with GPU option?
  if( clParser.isOptionTriggered("usingGpu") ){
#ifdef GUNDAM_USING_CACHE_MANAGER
    LogThrowIf( not Cache::Manager::HasCUDA(), "CUDA support not enabled with this GUNDAM build." );
#else
    LogThrow("CUDA support not enabled with this GUNDAM build (GUNDAM_USING_CACHE_MANAGER required).")
#endif
    LogWarning << "Using GPU parallelization." << std::endl;
  }

  if (clParser.isOptionTriggered("forceDirect")) GundamGlobals::setForceDirectCalculation(true);

  bool useCache = false;
#ifdef GUNDAM_USING_CACHE_MANAGER
  useCache = Cache::Manager::HasGPU(true);
#endif
  if (clParser.isOptionTriggered("usingCacheManager")) {
      int values = clParser.getNbValueSet("usingCacheManager");
      if (values < 1) useCache = not useCache;
      else if ("on" == clParser.getOptionVal<std::string>("usingCacheManager",0)) useCache = true;
      else if ("off" == clParser.getOptionVal<std::string>("usingCacheManager",0)) useCache = false;
      else {
          LogThrow("Invalid --cache-manager argument: must be empty, 'on' or 'off'");
      }
  }
  if (clParser.isOptionTriggered("usingGpu")) useCache = true;

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
  if( clParser.isOptionTriggered("noDialCache") ){
    LogAlert << "Disabling cache in dial evaluation (when available)..." << std::endl;
    GundamGlobals::setDisableDialCache(true);
  }

  // inject parameter config?
  std::string injectParameterPath{};
  if( clParser.isOptionTriggered("injectParameterConfig") ){
    LogWarning << "Inject parameter config: " << clParser.getOptionVal<std::string>("injectParameterConfig") << std::endl;
    injectParameterPath = clParser.getOptionVal<std::string>("injectParameterConfig");
  }

  // toy par injector
  std::string toyParInjector{};
  if( clParser.isOptionTriggered("injectToyParameters") ){
    toyParInjector = clParser.getOptionVal<std::string>("injectToyParameters");
    LogWarning << "Inject toy parameter: " << toyParInjector << std::endl;
  }

  // PRNG seed?
  gRandom = new TRandom3(0);    // Initialize with a UUID;
  if( clParser.isOptionTriggered("randomSeed") ){
    LogAlert << "Using user-specified random seed: " << clParser.getOptionVal<ULong_t>("randomSeed") << std::endl;
    gRandom->SetSeed(clParser.getOptionVal<ULong_t>("randomSeed"));
  }

  // How many parallel threads?
  GundamGlobals::setNumberOfThreads( clParser.getOptionVal("nbThreads", 1) );
  LogInfo << "Running the fitter with " << GundamGlobals::getNbCpuThreads() << " parallel threads." << std::endl;

  // Reading configuration
  auto configFilePath = clParser.getOptionVal("configFile", "");
  LogThrowIf(configFilePath.empty(), "Config file not provided.");

  ConfigUtils::ConfigHandler configHandler(configFilePath);
  configHandler.override( clParser.getOptionValList<std::string>("overrideFiles") );
  configHandler.flatOverride( clParser.getOptionValList<std::string>("overrides") );

  auto gundamFitterConfig(configHandler.getConfig());

  // Output file path
  std::string outFileName;
  if( clParser.isOptionTriggered("outputFilePath") ){
    outFileName = clParser.getOptionVal("outputFilePath", outFileName + ".root");
  }
  else{

    std::string outFolder{"./"};
    GenericToolbox::Json::fillValue(gundamFitterConfig, outFolder, "outputFolder");
    if( clParser.isOptionTriggered("outputDir") ){ outFolder = clParser.getOptionVal<std::string>("outputDir"); }

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
        "gundamFitter_" + GundamUtils::generateFileName(clParser, appendixDict)
    ) + ".root";
  }


  // --------------------------
  // Initialize the fitter:
  // --------------------------

  // Checking the minimal version for the config
  if( clParser.isOptionTriggered("ignoreVersionCheck") ){
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
  app.setCmdLinePtr( &clParser );

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
  if( clParser.isOptionTriggered("asimov") ){
    fitter.getLikelihoodInterface().setForceAsimovData( true );
    fitter.getLikelihoodInterface().setDataType( LikelihoodInterface::DataType::Asimov );
  }
  else{
    // by default, assume it's a real/fake data fit
    fitter.getLikelihoodInterface().setDataType( LikelihoodInterface::DataType::RealData );
  }


  // --use-data-entry
  if( clParser.isOptionTriggered("useDataEntry") ){
    auto selectedDataEntry = clParser.getOptionVal<std::string>("useDataEntry", 0);
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
  fitter.getMinimizer().setDisableCalcError( clParser.isOptionTriggered("skipHesse") );

  // --scan <N>
  if( clParser.isOptionTriggered("scanParameters") ) {
    fitter.setEnablePreFitScan( true );
    fitter.setEnablePostFitScan( true );
    fitter.getParameterScanner().setNbPoints(
        clParser.getOptionVal("scanParameters", fitter.getParameterScanner().getNbPoints())
    );
  }

  // --enable-pca
  if( clParser.isOptionTriggered("enablePca") ){
    fitter.setEnablePca( true );
    if( clParser.getNbValueSet("enablePca") == 1 ){
      fitter.setPcaThreshold( clParser.getOptionVal<double>("enablePca") );
    }
    if( clParser.getNbValueSet("enablePca") == 2 ){
      fitter.setPcaThreshold( clParser.getOptionVal<double>("enablePca", 0) );
      fitter.setPcaMethod( FitterEngine::PcaMethod::toEnum(clParser.getOptionVal<std::string>("enablePca", 1), true) );
    }
  }

  // --toy <iToy>
  if( clParser.isOptionTriggered("toyFit") ){
    fitter.getLikelihoodInterface().setDataType( LikelihoodInterface::DataType::Toy );
    fitter.getLikelihoodInterface().getModelPropagator().setIThrow( clParser.getOptionVal("toyFit", -1) );
  }

  // -d
  fitter.setIsDryRun( clParser.isOptionTriggered("dry-run") );

  // --one-sigma
  fitter.setGenerateOneSigmaPlots( clParser.isOptionTriggered("generateOneSigmaPlots") );

  // --light-mode
  GundamGlobals::setLightOutputMode(clParser.isOptionTriggered("lightOutputMode") );

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
  if (clParser.isOptionTriggered("kickMc")) {
      int values = clParser.getNbValueSet("usingCacheManager");
      if (values < 1) kickMc = 0.1; // Default without an argument.
      else kickMc = clParser.getOptionVal<double>("kickMc",0);
  }
  if ( kickMc > 0.01 ) {
      LogAlert << "Fit starting point randomized by " << kickMc << " sigma"
               << " around prior values."
               << std::endl;
      fitter.setThrowMcBeforeFit( true );
      fitter.setThrowGain( kickMc );
  }

  if( clParser.isOptionTriggered("debugMaxNbEventToLoad") ){
    LogThrowIf(clParser.getNbValueSet("debugMaxNbEventToLoad") != 1, "Nb of event not specified.");
    LogDebug << "Load " << clParser.getOptionVal<size_t>("debugMaxNbEventToLoad") << "max events per dataset." << std::endl;
    for( auto& dataset : fitter.getLikelihoodInterface().getDatasetList() ){
      dataset.setNbMaxEventToLoad(clParser.getOptionVal<size_t>("debugMaxNbEventToLoad"));
    }
  }

  // --------------------------
  // Load:
  // --------------------------
  fitter.initialize();

  // show initial conditions
  if( clParser.isOptionTriggered("injectParameterConfig") ) {
    LogDebug << "Starting mc parameters that where injected:" << std::endl;
    LogDebug << fitter.getLikelihoodInterface().getModelPropagator().getParametersManager().getParametersSummary(false ) << std::endl;
  }

  if( clParser.isOptionTriggered("scanLine") ){
    auto* outDir = GenericToolbox::mkdirTFile(fitter.getSaveDir(), GenericToolbox::joinPath("preFit", "cmdScanLine"));
    LogThrowIf( clParser.getNbValueSet("scanLine") == 0, "No injector file provided.");
    if( clParser.getNbValueSet("scanLine") == 1 ){
      LogAlert << "Will scan the line toward the point set in: " << clParser.getOptionVal<std::string>("scanLine", 0) << std::endl;

      auto endPoint = ConfigUtils::readConfigFile(clParser.getOptionVal<std::string>("scanLine", 0));

      GenericToolbox::writeInTFile( outDir, TNamed("endPoint", GenericToolbox::Json::toReadableString(endPoint).c_str()) );

      fitter.getParameterScanner().scanSegment( outDir, endPoint );
    }
    else if( clParser.getNbValueSet("scanLine") == 2 ) {
      LogAlert << "Will scan the line from point A (" << clParser.getOptionVal<std::string>("scanLine", 0)
          << ") to point B (" << clParser.getOptionVal<std::string>("scanLine", 1) << ")" << std::endl;

      auto startPoint = ConfigUtils::readConfigFile(clParser.getOptionVal<std::string>("scanLine", 0));
      auto endPoint = ConfigUtils::readConfigFile(clParser.getOptionVal<std::string>("scanLine", 1));

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
