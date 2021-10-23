//
// Created by Nadrino on 21/04/2021.
//

#include "string"

#include "Logger.h"
#include "CmdLineParser.h"

#include "JsonUtils.h"
#include "gundamFlatTreeConverter.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[FlatTreeConverter]");
})

int main( int argc, char** argv ){

  CmdLineParser clp(argc, argv);
  clp.addOption("config-file", {"-c"}, "Provide YAML/Json configuration file.", 1);
  clp.addOption("output-file-path", {"-o"}, "Set output file path.", 1);

  LogInfo << "Available options: " << std::endl;
  LogInfo << clp.getConfigSummary() << std::endl;

  clp.parseCmdLine();

  LogWarning << "Command line options:" << std::endl;
  LogWarning << clp.getValueSummary() << std::endl;

  auto configFilePath = clp.getOptionVal<std::string>("config-file");

  FlatTreeConverter ftc;
  ftc.loadConfig(JsonUtils::readConfigFile(configFilePath));

}

void FlatTreeConverter::loadConfig(const nlohmann::json &config_) {
  this->_config_ = config_;
}
