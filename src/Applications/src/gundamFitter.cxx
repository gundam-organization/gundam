//
// Created by Nadrino on 01/06/2021.
//

#include "FitterEngine.h"
#include "VersionConfig.h"
#include "JsonUtils.h"
#include "GlobalVariables.h"
#include "GundamGreetings.h"
#ifdef GUNDAM_USING_CACHE_MANAGER
#include "CacheManager.h"
#endif
#include "CmdLineParser.h"
#include "Logger.h"
#include "GenericToolbox.Root.h"

#include <string>


LoggerInit([]{
  Logger::setUserHeaderStr("[gundamFitter.cxx]");
});

int main(int argc, char** argv){

  GundamGreetings g;
  g.setAppName("GundamFitter");
  g.hello();

#ifdef GUNDAM_USING_CACHE_MANAGER
  if (Cache::Manager::HasCUDA()){ LogWarning << "CUDA compatible build." << std::endl; }
#endif

  // --------------------------
  // Read Command Line Args:
  // --------------------------
  CmdLineParser clParser;

  clParser.addDummyOption("Main options");

  clParser.addOption("configFile", {"-c", "--config-file"}, "Specify path to the fitter config file");
  clParser.addOption("nbThreads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");
  clParser.addOption("outputFile", {"-o", "--out-file"}, "Specify the output file");
  clParser.addOption("randomSeed", {"-s", "--seed"}, "Set random seed");
  clParser.addOption("appendix", {"--appendix"}, "Add appendix to the output file name");

  clParser.addDummyOption("Trigger options");

  clParser.addTriggerOption("dry-run", {"--dry-run", "-d"},"Perform the full sequence of initialization, but don't do the actual fit.");
  clParser.addTriggerOption("asimov", {"-a", "--asimov"}, "Use MC dataset to fill the data histograms");
  clParser.addTriggerOption("enablePca", {"--enable-pca"}, "Enable principle component analysis for eigen decomposed parameter sets");
  clParser.addTriggerOption("skipHesse", {"--skip-hesse"}, "Don't perform postfit error evaluation");
  clParser.addTriggerOption("generateOneSigmaPlots", {"--one-sigma"}, "Generate one sigma plots");
  clParser.addOption("scanParameters", {"--scan"}, "Enable parameter scan before and after the fit (can provide nSteps)", 1, true);
  clParser.addOption("toyFit", {"--toy"}, "Run a toy fit (optional arg to provide toy index)", 1, true);

  clParser.addDummyOption("Runtime/debug options");

  clParser.addOption("debugVerbose", {"--debug"}, "Enable debug verbose (can provide verbose level arg)", 1, true);
  clParser.addTriggerOption("usingCacheManager", {"--cache-manager"}, "Event weight cache handle by the CacheManager");
  clParser.addTriggerOption("usingGpu", {"--gpu"}, "Use GPU parallelization");

  clParser.addDummyOption();


  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogThrowIf(clParser.isNoOptionTriggered(), "No option was provided.");

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;
  LogInfo << clParser.dumpConfigAsJsonStr() << std::endl;

  if( clParser.isOptionTriggered("debugVerbose") ) GlobalVariables::setVerboseLevel(clParser.getOptionVal("debugVerbose", 1));

  bool useGpu = clParser.isOptionTriggered("usingGpu");
  if( useGpu ){
#ifdef GUNDAM_USING_CACHE_MANAGER
    LogThrowIf( not Cache::Manager::HasCUDA(), "CUDA support not enabled with this GUNDAM build." );
#else
    LogThrow("CUDA support not enabled with this GUNDAM build (GUNDAM_USING_CACHE_MANAGER required).")
#endif
    LogWarning << "Using GPU parallelization." << std::endl;
  }

  bool useCacheManager = clParser.isOptionTriggered("usingCacheManager") or useGpu;
  if( useCacheManager ){
#ifdef GUNDAM_USING_CACHE_MANAGER
    GlobalVariables::setEnableCacheManager(true);
#else
    LogThrow("useCacheManager can only be set while GUNDAM is compiled with GUNDAM_USING_CACHE_MANAGER option.");
#endif
  }

  if( clParser.isOptionTriggered("randomSeed") ){
    LogAlert << "Using user-specified random seed: " << clParser.getOptionVal<ULong_t>("randomSeed") << std::endl;
    gRandom->SetSeed(clParser.getOptionVal<ULong_t>("randomSeed"));
  }
  else{
    ULong_t seed = time(nullptr);
    LogInfo << "Using \"time(nullptr)\" random seed: " << seed << std::endl;
    gRandom->SetSeed(seed);
  }

  auto configFilePath = clParser.getOptionVal("configFile", "");
  LogThrowIf(configFilePath.empty(), "Config file not provided.");

  GlobalVariables::setNbThreads(clParser.getOptionVal("nbThreads", 1));
  LogInfo << "Running the fitter with " << GlobalVariables::getNbThreads() << " parallel threads." << std::endl;

  // --------------------------
  // Initialize the fitter:
  // --------------------------
  LogInfo << "Reading config file: " << configFilePath << std::endl;
  auto jsonConfig = JsonUtils::readConfigFile(configFilePath); // works with yaml

  if( JsonUtils::doKeyExist(jsonConfig, "minGundamVersion") ){
    LogThrowIf(
        not g.isNewerOrEqualVersion(JsonUtils::fetchValue<std::string>(jsonConfig, "minGundamVersion")),
        "Version check FAILED: " << GundamVersionConfig::getVersionStr() << " < " << JsonUtils::fetchValue<std::string>(jsonConfig, "minGundamVersion")
    );
    LogInfo << "Version check passed: " << GundamVersionConfig::getVersionStr() << " >= " << JsonUtils::fetchValue<std::string>(jsonConfig, "minGundamVersion") << std::endl;
  }

  bool isDryRun = clParser.isOptionTriggered("dry-run");
  bool enableParameterScan = clParser.isOptionTriggered("scanParameters") or JsonUtils::fetchValue(jsonConfig, "scanParameters", false);
  int nbScanSteps = clParser.getOptionVal("scanParameters", 100);

  bool isToyFit = clParser.isOptionTriggered("toyFit");
  int iToyFit = clParser.getOptionVal("toyFit", -1);

  std::string outFileName;
  if( clParser.isOptionTriggered("outputFile") ){
    outFileName = clParser.getOptionVal("outputFile", outFileName + ".root");
  }
  else{
    if( JsonUtils::doKeyExist(jsonConfig, "outputFolder") ){
      GenericToolbox::mkdirPath(JsonUtils::fetchValue<std::string>(jsonConfig, "outputFolder"));
      outFileName = JsonUtils::fetchValue<std::string>(jsonConfig, "outputFolder");
      outFileName += "/";
    }
    outFileName += GenericToolbox::getFileNameFromFilePath(configFilePath, false);

    if( clParser.isOptionTriggered("asimov") ){ outFileName += "_Asimov"; }
    if( clParser.isOptionTriggered("scanParameters") ){ outFileName += "_Scan"; }
    if( clParser.isOptionTriggered("generateOneSigmaPlots") ){ outFileName += "_OneSigma"; }
    if( clParser.isOptionTriggered("enablePca") ){ outFileName += "_PCA"; }
    if( clParser.isOptionTriggered("skipHesse") ){ outFileName += "_NoHesse"; }
    if( clParser.isOptionTriggered("toyFit") ){
      outFileName += "_toyFit";
      if( iToyFit != -1 ){ outFileName += "_" + std::to_string(iToyFit); }
    }
    if( clParser.isOptionTriggered("dry-run") ){ outFileName += "_DryRun"; }
    if( clParser.isOptionTriggered("appendix") ){ outFileName += "_" + clParser.getOptionVal<std::string>("appendix"); }

    outFileName += ".root";
  }

  LogWarning << "Creating output file: \"" << outFileName << "\"..." << std::endl;
  TFile* out = TFile::Open(outFileName.c_str(), "RECREATE");


  LogInfo << "Writing runtime parameters in output file..." << std::endl;

  // Gundam version?
  TNamed gundamVersionString("gundamVersion", GundamVersionConfig::getVersionStr().c_str());
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(out, "gundamFitter"), &gundamVersionString);

  // Command line?
  TNamed commandLineString("commandLine", clParser.getCommandLineString().c_str());
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(out, "gundamFitter"), &commandLineString);

  // Config unfolded ?
  auto unfoldedConfig = jsonConfig;
  JsonUtils::unfoldConfig(unfoldedConfig);
  std::stringstream ss;
  ss << unfoldedConfig << std::endl;
  TNamed unfoldedConfigString("unfoldedConfig", ss.str().c_str());
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(out, "gundamFitter"), &unfoldedConfigString);


  LogInfo << "FitterEngine setup..." << std::endl;

  // Fitter
  FitterEngine fitter;
  fitter.setConfig(JsonUtils::fetchSubEntry(jsonConfig, {"fitterEngineConfig"}));
  fitter.setSaveDir(GenericToolbox::mkdirTFile(out, "FitterEngine"));
  fitter.setNbScanSteps(nbScanSteps);
  fitter.setEnablePostFitScan(enableParameterScan);
  fitter.setEnablePca(clParser.isOptionTriggered("enablePca"));

  if( isToyFit ){
    fitter.getPropagator().setThrowAsimovToyParameters(true);
    fitter.getPropagator().setIThrow(iToyFit);
  }
  fitter.getPropagator().setLoadAsimovData( clParser.isOptionTriggered("asimov") );

  fitter.initialize();

  if( clParser.isOptionTriggered("skipHesse") ) fitter.setEnablePostFitErrorEval(false);

  fitter.updateChi2Cache();
  LogInfo << "Initial χ² = " << fitter.getChi2Buffer() << std::endl;
  LogInfo << "Initial χ²(stat) = " << fitter.getChi2StatBuffer() << std::endl;

  // --------------------------
  // Pre-fit:
  // --------------------------

  // Event rates variations
  if( JsonUtils::doKeyExist(jsonConfig, "allParamVariations") )
  {
    fitter.varyEvenRates(JsonUtils::fetchValue<std::vector<double>>(jsonConfig,
                                                                    "allParamVariations",
                                                                    std::vector<double>()),
                         "preFit");
  }

  // LLH Visual Scan
  if( clParser.isOptionTriggered("generateOneSigmaPlots") or JsonUtils::fetchValue(jsonConfig, "generateOneSigmaPlots", false) ) fitter.generateOneSigmaPlots("preFit");
  if( clParser.isOptionTriggered("scanParameters") or JsonUtils::fetchValue(jsonConfig, "scanParameters", false) ) fitter.scanParameters(nbScanSteps, "preFit/scan");

  // Plot generators
  if( JsonUtils::fetchValue(jsonConfig, "generateSamplePlots", true) ) fitter.generateSamplePlots("preFit/samples");


  // --------------------------
  // Run the fitter:
  // --------------------------
  if( not isDryRun and JsonUtils::fetchValue(jsonConfig, "fit", true) ){
    fitter.fit();
  }

  LogWarning << "Closing output file \"" << out->GetName() << "\"..." << std::endl;
  out->Close();
  LogInfo << "Closed." << std::endl;

  // --------------------------
  // Goodbye:
  // --------------------------
  g.goodbye();

  GlobalVariables::getParallelWorker().reset();
}
