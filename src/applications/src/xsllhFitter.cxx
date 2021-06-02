//
// Created by Adrien BLANCHET on 01/06/2021.
//

#include "string"

#include "CmdLineParser.h"
#include "Logger.h"

#include "JsonUtils.h"
#include "FitParameterSet.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[xsllhFitter.cxx]");
} )

int main(int argc, char** argv){
  CmdLineParser clParser;

  clParser.addTriggerOption("dry-run", {"--dry-run", "-d"},"Perform the full sequence of initialization, but don't do the actual fit.");
  clParser.addOption("config-file", {"-c", "--config-file"}, "Specify path to the fitter config file");

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;

  auto configFilePath = clParser.getOptionVal<std::string>("config-file");
  auto jsonConfig = JsonUtils::readConfigFile(configFilePath);

  FitParameterSet parameterSet;
  parameterSet.setJsonConfig(jsonConfig["fitParameterSets"][0]);
  parameterSet.initialize();
  LogInfo << parameterSet.getSummary() << std::endl;

  FitParameterSet parameterSetXsec;
  parameterSetXsec.setJsonConfig(jsonConfig["fitParameterSets"][1]);
  parameterSetXsec.initialize();
  LogInfo << parameterSetXsec.getSummary() << std::endl;

}