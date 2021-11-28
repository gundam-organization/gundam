//
// Created by Nadrino on 01/06/2021.
//

#include "string"

#include "yaml-cpp/yaml.h"

#include "CmdLineParser.h"
#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"

#include "JsonUtils.h"
#include "Propagator.h"
#include "FitterEngine.h"
#include "GlobalVariables.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[gundamFitter.cxx]");
} )

int main(int argc, char** argv){


  ///////////////////////////////
  // Read Command Line Args:
  /////////////////////////////
  CmdLineParser clParser;

  clParser.addTriggerOption("dry-run", {"--dry-run", "-d"},"Perform the full sequence of initialization, but don't do the actual fit.");
  clParser.addTriggerOption("generateOneSigmaPlots", {"--one-sigma"}, "Generate one sigma plots");

  clParser.addOption("configFile", {"-c", "--config-file"}, "Specify path to the fitter config file");
  clParser.addOption("nbThreads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");
  clParser.addOption("outputFile", {"-o", "--out-file"}, "Specify the output file");
  clParser.addOption("scanParameters", {"--scan"}, "Enable parameter scan before and after the fit");

  clParser.getOptionPtr("scanParameters")->setAllowEmptyValue(true); // --scan can be followed or not by the number of steps

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogThrowIf(clParser.isNoOptionTriggered(), "No option was provided.");

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;
  LogInfo << clParser.dumpConfigAsJsonStr() << std::endl;

  auto configFilePath = clParser.getOptionVal("configFile", "");
  LogThrowIf(configFilePath.empty(), "Config file not provided.");

  GlobalVariables::setNbThreads(clParser.getOptionVal("nbThreads", 1));
  LogInfo << "Running the fitter with " << GlobalVariables::getNbThreads() << " parallel threads." << std::endl;

  bool isDryRun = clParser.isOptionTriggered("dry-run");
  bool enableParameterScan = clParser.isOptionTriggered("scanParameters");
  int nbScanSteps = clParser.getOptionVal("scanParameters", 100);
  auto outFileName = clParser.getOptionVal("outputFile", configFilePath + ".root");

  ///////////////////////////////
  // Initialize the fitter:
  /////////////////////////////
  LogInfo << "Reading config file: " << configFilePath << std::endl;
  auto jsonConfig = JsonUtils::readConfigFile(configFilePath); // works with yaml

  LogWarning << "Creating output file: \"" << outFileName << "\"" << std::endl;
  TFile* out = TFile::Open(outFileName.c_str(), "RECREATE");

  FitterEngine fitter;
  fitter.setConfig(JsonUtils::fetchSubEntry(jsonConfig, {"fitterEngineConfig"}));
  fitter.setSaveDir(GenericToolbox::mkdirTFile(out, "FitterEngine"));
  fitter.setNbScanSteps(nbScanSteps);
  fitter.setEnablePostFitScan(enableParameterScan);
  fitter.initialize();

  ///////////////////////////////
  // Pre-fit:
  /////////////////////////////

  // LLH Visual Scan
  if( clParser.isOptionTriggered("generateOneSigmaPlots") or JsonUtils::fetchValue(jsonConfig, "generateOneSigmaPlots", false) ) fitter.generateOneSigmaPlots("preFit");
  if( clParser.isOptionTriggered("scanParameters") or JsonUtils::fetchValue(jsonConfig, "scanParameters", true) ) fitter.scanParameters(nbScanSteps, "preFit/scan");

  // State before the fit
  if( JsonUtils::fetchValue(jsonConfig, "throwMcBeforeFit", false) ){
    LogInfo << "Throwing parameters on Pre-fit MC..." << std::endl;
    fitter.throwParameters( JsonUtils::fetchValue(jsonConfig, "throwMcBeforeFitGain", 1.) );
    fitter.updateChi2Cache();
    LogInfo << "Chi2 stat after MC thrown: " << fitter.getChi2StatBuffer() << std::endl;
  }
  if( JsonUtils::fetchValue(jsonConfig, "generateSamplePlots", true) ) fitter.generateSamplePlots("preFit/samples");

  ///////////////////////////////
  // Run the fitter:
  /////////////////////////////
  if( not isDryRun and JsonUtils::fetchValue(jsonConfig, "fit", true) ){
    fitter.fit();
    if( fitter.isFitHasConverged() ) fitter.writePostFitData();
  }

  LogWarning << "Closing output file \"" << out->GetName() << "\"..." << std::endl;
  out->Close();
  LogInfo << "Closed." << std::endl;

}