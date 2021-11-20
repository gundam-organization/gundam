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

  clParser.addOption("config-file", {"-c", "--config-file"}, "Specify path to the fitter config file");
  clParser.addOption("nb-threads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");
  clParser.addOption("output-file", {"-o", "--out-file"}, "Specify the output file");

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;
  LogInfo << clParser.dumpConfigAsJsonStr() << std::endl;

  auto configFilePath = clParser.getOptionVal<std::string>("config-file");

  int nThreads = 1;
  if( clParser.isOptionTriggered("nb-threads") ) nThreads = clParser.getOptionVal<int>("nb-threads");
  GlobalVariables::setNbThreads(nThreads);

  bool isDryRun = clParser.isOptionTriggered("dry-run");

  std::string outFileName = configFilePath + ".root";
  if( clParser.isOptionTriggered("output-file") ) outFileName = clParser.getOptionVal<std::string>("output-file");

  ///////////////////////////////
  // Initialize the fitter:
  /////////////////////////////
  LogInfo << "Reading config file: " << configFilePath << std::endl;
  auto jsonConfig = JsonUtils::readConfigFile(configFilePath); // works with yaml

  TFile* out = TFile::Open(outFileName.c_str(), "RECREATE");

  FitterEngine fitter;
  fitter.setConfig(JsonUtils::fetchSubEntry(jsonConfig, {"fitterEngineConfig"}));
  fitter.setSaveDir(GenericToolbox::mkdirTFile(out, "FitterEngine"));
  fitter.initialize();

  ///////////////////////////////
  // Pre-fit:
  /////////////////////////////

  // LLH Visual Scan
  if( clParser.isOptionTriggered("generateOneSigmaPlots") or JsonUtils::fetchValue(jsonConfig, "generateOneSigmaPlots", false) ) fitter.generateOneSigmaPlots("preFit");
  if( JsonUtils::fetchValue(jsonConfig, "scanParameters", true) ) fitter.scanParameters(10, "preFit/scan");

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
    if( JsonUtils::fetchValue(jsonConfig, "scanParameters", true) ) fitter.scanParameters(10, "postFit/scan");
  }

  LogWarning << "Closing output file \"" << out->GetName() << "\"..." << std::endl;
  out->Close();
  LogInfo << "Closed." << std::endl;

}