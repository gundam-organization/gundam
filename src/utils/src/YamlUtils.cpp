//
// Created by Adrien BLANCHET on 17/06/2021.
//

#include "Logger.h"

#include "GenericToolbox.h"
#include "YamlUtils.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[YamlUtils]");
})


YAML::Node YamlUtils::readConfigFile(const std::string &configFilePath_) {

  if( not GenericToolbox::doesPathIsFile(configFilePath_) ){
    LogError << "\"" << configFilePath_ << "\" could not be found." << std::endl;
    throw std::runtime_error("file not found.");
  }

  return YAML::LoadFile(configFilePath_);
}
