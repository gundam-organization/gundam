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


int main(int argc, char** argv){

  GundamApp app{"main fitter"};

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

  clParser.addDummyOption("Input options");
  clParser.addOption("configFile", {"-c", "--config-file"}, "Specify path to the fitter config file");
  clParser.addOption("overrideFiles", {"-of", "--override-files"}, "Provide config files that will override keys", -1);
  clParser.addOption("overrides", {"-O", "--override"}, "Add a command line override [e.g. /fitterEngineConfig/engineType=mcmc)", -1);
  clParser.addDummyOption();

  clParser.addDummyOption("Runtime options");
  clParser.addOption("nbThreads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");
  clParser.addOption("randomSeed", {"-s", "--seed"}, "Set random seed");
  clParser.addOption("enablePca", {"--pca", "--enable-pca"}, "Enable principle component analysis for eigen decomposed parameter sets", 2, true);
  clParser.addDummyOption();

  clParser.addDummyOption("Fit options");
  clParser.addTriggerOption("asimov", {"-a", "--asimov"}, "Use MC dataset to fill the data histograms");
  clParser.addTriggerOption("skipHesse", {"--skip-hesse"}, "Don't perform postfit error evaluation");
  clParser.addOption("toyFit", {"--toy"}, "Run a toy fit (optional arg to provide toy index)", 1, true);
  clParser.addOption("injectParameterConfig", {"--inject-parameters"}, "Inject parameters defined in the provided config file");
  clParser.addOption("injectToyParameters", {"--inject-toy-parameter"}, "Inject parameters defined in the provided config file");
  clParser.addDummyOption();

  clParser.addDummyOption("Output options");
  clParser.addOption("outputFilePath", {"-o", "--out-file"}, "Specify the output file");
  clParser.addOption("outputDir", {"--out-dir"}, "Specify the output directory");
  clParser.addOption("appendix", {"--appendix"}, "Add appendix to the output file name");
  clParser.addOption("scanParameters", {"--scan"}, "Enable parameter scan before and after the fit (can provide nSteps)", 1, true);
  clParser.addOption("scanLine", {"--scan-line"}, "Provide par injector files: start and end point or only end point (start will be prefit)", 2, true);
  clParser.addTriggerOption("generateOneSigmaPlots", {"--one-sigma"}, "Generate one sigma plots");
  clParser.addTriggerOption("lightOutputMode", {"--light-mode"}, "Disable plot generation");
  clParser.addDummyOption();

  clParser.addDummyOption("Debug options");
  clParser.addTriggerOption("dry-run", {"-d", "--dry-run"},"Perform the full sequence of initialization, but don't do the actual fit.");
  clParser.addTriggerOption("super-dry-run", {"-dd", "--super-dry-run"},"Only reads the config files.");
  clParser.addOption("debugVerbose", {"--debug"}, "Enable debug verbose (can provide verbose level arg)", 1, true);
  clParser.addOption("kickMc", {"--kick-mc"}, "Amount to push the starting parameters away from their prior values (default: 0)", 1, true);
  clParser.addTriggerOption("ignoreVersionCheck", {"--ignore-version"}, "Don't check GUNDAM version with config request");
  clParser.addOption("debugMaxNbEventToLoad", {"-me", "--max-events"}, "Set the maximum number of events to load per dataset", 1);
  clParser.addOption("debugFracOfEntries", {"-fe", "--fraction-of-entries"}, "Set the fraction of the total entries of each TTree that will be read", 1);
  clParser.addDummyOption();

#ifdef GUNDAM_USING_CACHE_MANAGER
  LogInfoIf( Cache::Manager::HasCUDA() ) << "CUDA inside." << std::endl;

  clParser.addDummyOption("GPU/CacheManager options");
  clParser.addTriggerOption("usingGpu", {"--gpu"}, "Use GPU parallelization (will enable the CacheManager)");
  clParser.addOption("usingCacheManager", {"--cache-manager"}, "Toggle the usage of the CacheManager (i.e. the GPU) [empty, 'on', or 'off']",1,true);
  clParser.addTriggerOption("forceDirect", {"--cpu"}, "Force direct calculation of weights (for debugging)");
#else
  clParser.addDummyOption("GPU/CacheManager options disabled. Needs to compile using -D WITH_CACHE_MANAGER");
#endif

  LogInfo << clParser.getDescription().str() << std::endl;

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogExitIf(clParser.isNoOptionTriggered(), "No option was provided.");

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;
  LogInfo << clParser.dumpConfigAsJsonStr() << std::endl;


  // --------------------------
  // Init command line args:
  // --------------------------
  GundamGlobals::setIsDebug( clParser.isOptionTriggered("debugVerbose") );

#ifdef GUNDAM_USING_CACHE_MANAGER
  bool hasGPU{Cache::Manager::HasGPU(true)};
  bool useCacheManager{hasGPU};

  // Is build compatible with GPU option?
  if( clParser.isOptionTriggered("usingGpu") ){
    LogExitIf(not hasGPU, "Option --gpu set, but no GPU is available");
    LogWarning << "Using GPU parallelization." << std::endl;
    useCacheManager = true;
  }

  if( clParser.isOptionTriggered("usingCacheManager") ){
    if( clParser.getNbValueSet("usingCacheManager") == 0 ){
      useCacheManager = true;
    }
    else{
      // value specified explicitly by the user
      auto useCacheManagerCli = clParser.getOptionVal<std::string>("usingCacheManager");

      if( useCacheManagerCli == "on" ){ useCacheManager = true; }
      else if( useCacheManagerCli == "off" ){ useCacheManager = false; }
      else{
        LogError << "Invalid --cache-manager argument: \""
                 << useCacheManagerCli << "\""
                 << std::endl;
        LogExit("Invalid --cache-manager argument: must be empty, 'on' or 'off'"); }
    }
  }

  LogWarningIf(hasGPU && not useCacheManager)
      << "A GPU is available, but not being used"
      << std::endl;

  // decision has been made
  Cache::Manager::SetIsEnabled( useCacheManager );

  Cache::Manager::SetIsForceCpuCalculation( clParser.isOptionTriggered("forceDirect") );
  Cache::Manager::SetEnableDebugPrintouts( clParser.isOptionTriggered("debugVerbose") );
#endif

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
  LogExitIf(configFilePath.empty(), "Config file not provided.");

  ConfigUtils::ConfigHandler configHandler(configFilePath);
  configHandler.override( clParser.getOptionValList<std::string>("overrideFiles") );
  configHandler.flatOverride( clParser.getOptionValList<std::string>("overrides") );

  auto gundamFitterConfig(configHandler.getConfig());

  // All of the fields that should (or may) be at this level in the YAML.
  // This provides a rudimentary syntax check for user inputs.
  ConfigUtils::checkFields(gundamFitterConfig,
                           "TOP LEVEL",
                           // Allowed fields (don't need to list fields in
                           // expected, or deprecated).
                           {{"outputFolder"},
                            {"minGundamVersion"},
                           },
                           // Expected fields (must be present)
                           {{"fitterEngineConfig"},
                           },
                           // Deprecated fields (allowed, but cause a warning)
                           {{"generateSamplePlots"},
                            {"allParameterVariations"},
                           },
                           // Replaced field (allowed, but cause a warning)
                           {});

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
    LogExitIf(
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

  if( clParser.isOptionTriggered("super-dry-run") ) {
    LogInfo << "Super-dry-run enabled. Stopping here." << std::endl;
    std::exit(EXIT_SUCCESS);
  }


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
  if (clParser.isOptionTriggered("kickMc")) {

    double kickMc = 0.1;
    if( clParser.getNbValueSet("kickMc") == 1 ){
      kickMc = clParser.getOptionVal<double>("kickMc");
    }

    LogAlert << "Fit starting point randomized by " << kickMc << " sigma"
             << " around prior values."
             << std::endl;
    fitter.setThrowMcBeforeFit( true );
    fitter.setThrowGain( kickMc );

  }

  if( clParser.isOptionTriggered("debugMaxNbEventToLoad") ){
    LogExitIf(clParser.getNbValueSet("debugMaxNbEventToLoad") != 1, "Nb of event not specified.");
    LogAlert << "Will load " << clParser.getOptionVal<size_t>("debugMaxNbEventToLoad") << " events per dataset." << std::endl;
    for( auto& dataset : fitter.getLikelihoodInterface().getDatasetList() ){
      dataset.setNbMaxEventToLoad(clParser.getOptionVal<size_t>("debugMaxNbEventToLoad"));
    }
  }

  if( clParser.isOptionTriggered("debugFracOfEntries") ){
    LogExitIf(clParser.getNbValueSet("debugFracOfEntries") != 1, "Nb of event not specified.");

    auto fractionOfEntries{clParser.getOptionVal<double>("debugFracOfEntries")};
    LogExitIf(fractionOfEntries > 1, "fractionOfEntries should be between 0 and 1");
    LogExitIf(fractionOfEntries < 0, "fractionOfEntries should be between 0 and 1");

    LogDebug << "Will load " << fractionOfEntries*100. << "% of the datasets." << std::endl;
    for( auto& dataset : fitter.getLikelihoodInterface().getDatasetList() ){
      dataset.setFractionOfEntriesToLoad(fractionOfEntries);
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
    LogExitIf( clParser.getNbValueSet("scanLine") == 0, "No injector file provided.");
    if( clParser.getNbValueSet("scanLine") == 1 ){
      LogAlert << "Will scan the line toward the point set in: " << clParser.getOptionVal<std::string>("scanLine", 0) << std::endl;

      auto endPoint = ConfigUtils::readConfigFile(clParser.getOptionVal<std::string>("scanLine", 0));

      GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed("endPoint", GenericToolbox::Json::toReadableString(endPoint).c_str()) );

      fitter.getParameterScanner().scanSegment( outDir, endPoint );
    }
    else if( clParser.getNbValueSet("scanLine") == 2 ) {
      LogAlert << "Will scan the line from point A (" << clParser.getOptionVal<std::string>("scanLine", 0)
          << ") to point B (" << clParser.getOptionVal<std::string>("scanLine", 1) << ")" << std::endl;

      auto startPoint = ConfigUtils::readConfigFile(clParser.getOptionVal<std::string>("scanLine", 0));
      auto endPoint = ConfigUtils::readConfigFile(clParser.getOptionVal<std::string>("scanLine", 1));

      GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed("startPoint", GenericToolbox::Json::toReadableString(startPoint).c_str()) );
      GenericToolbox::writeInTFileWithObjTypeExt( outDir, TNamed("endPoint", GenericToolbox::Json::toReadableString(endPoint).c_str()) );

      fitter.getParameterScanner().scanSegment( outDir, endPoint, startPoint );
    }
    else{
      LogExitIf("Invalid --scan-line arguments");
    }
  }

  // --------------------------
  // Run the fitter:
  // --------------------------
  fitter.fit();

}
