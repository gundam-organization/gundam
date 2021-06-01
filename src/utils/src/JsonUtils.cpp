//
// Created by Adrien BLANCHET on 22/05/2021.
//

#include "stdexcept"

#include "GenericToolbox.h"
#include "Logger.h"

#include "JsonUtils.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[JsonUtils]");
} )

nlohmann::json JsonUtils::readConfigFile(const std::string& configFilePath_){

  if( not GenericToolbox::doesPathIsFile(configFilePath_) ){
    LogError << "\"" << configFilePath_ << "\" could not be found." << std::endl;
    throw std::runtime_error("file not found.");
  }

  std::fstream fs;
  fs.open(configFilePath_, std::ios::in);

  if( not fs.is_open() ) {
    LogError << "\"" << configFilePath_ << "\": could not read file." << std::endl;
    throw std::runtime_error("file not readable.");
  }

  nlohmann::json output;
  fs >> output;

  return output;
}
