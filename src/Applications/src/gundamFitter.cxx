//
// Created by Nadrino on 01/06/2021.
//

#include "versionConfig.h"
#include "FitterEngine.h"
#include "JsonUtils.h"
#include "GlobalVariables.h"
#include "GundamGreetings.h"

#include "CmdLineParser.h"
#include "Logger.h"
#include "GenericToolbox.Root.h"

#include <string>


LoggerInit([](){
  Logger::setUserHeaderStr("[gundamFitter.cxx]");
} )

int main(int argc, char** argv){

  GundamGreetings g;
  g.setAppName("GundamFitter");
  g.hello();


  // --------------------------
  // Read Command Line Args:
  // --------------------------
  CmdLineParser clParser;

  clParser.addTriggerOption("dry-run", {"--dry-run", "-d"},"Perform the full sequence of initialization, but don't do the actual fit.");
  clParser.addTriggerOption("cache", {"-C", "--no-cache"}, "Disable the event weight cache");
  clParser.addTriggerOption("generateOneSigmaPlots", {"--one-sigma"}, "Generate one sigma plots");

  clParser.addOption("configFile", {"-c", "--config-file"}, "Specify path to the fitter config file");
  clParser.addOption("nbThreads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");
  clParser.addOption("outputFile", {"-o", "--out-file"}, "Specify the output file");
  clParser.addOption("scanParameters", {"--scan"}, "Enable parameter scan before and after the fit");
  clParser.addOption("toyFit", {"--toy"}, "Run a toy fit");
  clParser.addOption("randomSeed", {"-s", "--seed"}, "Set random seed");

  clParser.getOptionPtr("scanParameters")->setAllowEmptyValue(true); // --scan can be followed or not by the number of steps
  clParser.getOptionPtr("toyFit")->setAllowEmptyValue(true); // --toy can be followed or not by the number of steps

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogThrowIf(clParser.isNoOptionTriggered(), "No option was provided.");

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;
  LogInfo << clParser.dumpConfigAsJsonStr() << std::endl;

  if( clParser.isOptionTriggered("randomSeed") ){
    LogAlert << "Using user-specified random seed: " << clParser.getOptionVal<ULong_t>("randomSeed") << std::endl;
    gRandom->SetSeed(clParser.getOptionVal<ULong_t>("randomSeed"));
  }
  else{
    ULong_t seed = time(nullptr);
    LogInfo << "Using \"time(nullptr)\" random seed: " << seed << std::endl;
    gRandom->SetSeed(seed);
  }

  if (clParser.isOptionTriggered("cache")) {
      LogInfo << "Event weight cache is disabled" << std::endl;
      GlobalVariables::setEnableEventWeightCache(false);
  }
  else {
      LogInfo << "Event weight cache is enabled" << std::endl;
      GlobalVariables::setEnableEventWeightCache(true);
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

  bool isDryRun = clParser.isOptionTriggered("dry-run");
  bool enableParameterScan = clParser.isOptionTriggered("scanParameters") or JsonUtils::fetchValue(jsonConfig, "scanParameters", false);
  int nbScanSteps = clParser.getOptionVal("scanParameters", 100);

  bool isToyFit = clParser.isOptionTriggered("toyFit");
  int iToyFit = clParser.getOptionVal("toyFit", -1);

  std::string outFileName = configFilePath;
  if( isToyFit ){
    outFileName += "_toyFit";
    if( iToyFit != -1 ){
      outFileName += "_" + std::to_string(iToyFit);
    }
  }
  outFileName = clParser.getOptionVal("outputFile", outFileName + ".root");
  LogWarning << "Creating output file: \"" << outFileName << "\"..." << std::endl;
  TFile* out = TFile::Open(outFileName.c_str(), "RECREATE");


  LogInfo << "Writing runtime parameters in output file..." << std::endl;

  // Gundam version?
  TNamed gundamVersionString("gundamVersion", getVersionStr().c_str());
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

  if( isToyFit ){
    fitter.getPropagator().setThrowAsimovToyParameters(true);
    fitter.getPropagator().setIThrow(iToyFit);
  }

  fitter.initialize();

  fitter.updateChi2Cache();
  LogInfo << "Initial χ² = " << fitter.getChi2Buffer() << std::endl;
  LogInfo << "Initial χ²(stat) = " << fitter.getChi2StatBuffer() << std::endl;

  // --------------------------
  // Pre-fit:
  // --------------------------

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

}
