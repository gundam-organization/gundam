//
// Created by Nadrino on 21/04/2021.
//

#include "string"

#include "Logger.h"
#include "CmdLineParser.h"

#include "JsonUtils.h"
#include "gundamFlatTreeConverter.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[gundamFlatTreeConverter.cxx]");
})

int main( int argc, char** argv ){

  CmdLineParser clp(argc, argv);
  clp.addOption("config-file", {"-c", "--config"}, "Provide YAML/Json configuration file.", 1);
  clp.parseCmdLine();

  auto configFilePath = clp.getOptionVal<std::string>("config-file");

  FlatTreeConverter ftc;
  ftc.loadConfig(JsonUtils::readConfigFile(configFilePath));

}

void FlatTreeConverter::loadConfig(const nlohmann::json &config_) {
  this->_config_ = config_;
}
