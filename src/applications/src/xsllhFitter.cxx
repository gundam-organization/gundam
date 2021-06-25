//
// Created by Adrien BLANCHET on 01/06/2021.
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
  Logger::setUserHeaderStr("[xsllhFitter.cxx]");
} )

int main(int argc, char** argv){
  CmdLineParser clParser;

  clParser.addTriggerOption("dry-run", {"--dry-run", "-d"},"Perform the full sequence of initialization, but don't do the actual fit.");
  clParser.addOption("config-file", {"-c", "--config-file"}, "Specify path to the fitter config file");
  clParser.addOption("nb-threads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");

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

  LogInfo << "Reading config file: " << configFilePath << std::endl;
  auto jsonConfig = JsonUtils::readConfigFile(configFilePath); // works with yaml

  TFile* out = TFile::Open("outTest.root", "RECREATE");

  FitterEngine fitter;
  fitter.setConfig(jsonConfig);
  fitter.setSaveDir(GenericToolbox::mkdirTFile(out, "fitter"));
  fitter.initialize();

  fitter.generateSamplePlots("prefit");
  fitter.generateOneSigmaPlots();

  LogDebug << "Closing output file: " << out->GetName() << std::endl;
  out->Close();
  LogDebug << "Closed." << std::endl;

}