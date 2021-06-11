//
// Created by Adrien BLANCHET on 01/06/2021.
//

#include "string"

#include "yaml-cpp/yaml.h"

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
  LogInfo << clParser.dumpConfigAsJsonStr() << std::endl;

  auto configFilePath = clParser.getOptionVal<std::string>("config-file");
  auto jsonConfig = JsonUtils::readConfigFile(configFilePath);

  YAML::Node node = YAML::LoadFile(configFilePath);
  std::cout << "YAML DUMP: " << YAML::Dump(node) << std::endl;

  std::vector<FitParameterSet> parameterSetList;
  for( const auto& parameterSetConfig : jsonConfig["fitParameterSets"] ){
    LogTrace << "emplace_back..." << std::endl;
    parameterSetList.emplace_back();
    LogTrace << "emplace_back... END" << std::endl;
    parameterSetList.back().setJsonConfig(parameterSetConfig);
    parameterSetList.back().initialize();
  }

  for( const auto& parameterSet : parameterSetList ){
    LogInfo << parameterSet.getSummary() << std::endl;
  }

}