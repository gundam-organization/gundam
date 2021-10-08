//
// Created by Nadrino on 17/06/2021.
//

#include "string"
#include "vector"

#include "yaml-cpp/yaml.h"

#include "CmdLineParser.h"
#include "Logger.h"
#include "GenericToolbox.h"

#include "YamlUtils.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[YamlSandbox]");
})


int main(int argc, char** argv){
  CmdLineParser clParser;

  clParser.addOption("config-file", {"-c", "--config-file"}, "Specify path to the fitter config file");

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;
  LogInfo << clParser.dumpConfigAsYamlStr() << std::endl;

  auto configFilePath = clParser.getOptionVal<std::string>("config-file");

  auto n = YamlUtils::readConfigFile(configFilePath);

  LogTrace << YAML::Dump(n) << std::endl;

  LogDebug << GET_VAR_NAME_VALUE(n.IsSequence()) << std::endl;
  LogDebug << GET_VAR_NAME_VALUE(n.IsMap()) << std::endl;

  auto fitParameterSets = YamlUtils::fetchValue<YAML::Node>(n, "fitParameterSets");
  LogTrace << YAML::Dump(fitParameterSets) << std::endl;

  LogDebug << GET_VAR_NAME_VALUE(fitParameterSets.IsSequence()) << std::endl;
  LogDebug << GET_VAR_NAME_VALUE(fitParameterSets.IsMap()) << std::endl;

  auto fluxSyst = YamlUtils::fetchMatchingEntry(fitParameterSets, "name", "Flux Systematics");
  LogTrace << YAML::Dump(fluxSyst) << std::endl;

  auto covarianceMatrixFilePath = YamlUtils::fetchValue(fluxSyst, "covarianceMatrixFilePath", "");
  LogTrace << GET_VAR_NAME_VALUE(covarianceMatrixFilePath) << std::endl;

  LogInfo << YamlUtils::toJson(n).dump() << std::endl;

}


