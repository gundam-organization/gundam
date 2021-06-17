//
// Created by Adrien BLANCHET on 22/05/2021.
//

#include "stdexcept"

#include "yaml-cpp/yaml.h"

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

  nlohmann::json output;

  if( GenericToolbox::doesFilePathHasExtension(configFilePath_, "yml") or GenericToolbox::doesFilePathHasExtension(configFilePath_,"yaml") ){
    LogDebug << "YAML input file detected" << std::endl;

    YAML::Node node = YAML::LoadFile(configFilePath_);
    YAML::Emitter emitter;
    emitter << YAML::DoubleQuoted << YAML::Flow << YAML::BeginSeq << node;
    std::string asJsonStr = std::string(emitter.c_str() + 1);
    GenericToolbox::replaceSubstringInsideInputString(asJsonStr, "\"true\"", "true");
    GenericToolbox::replaceSubstringInsideInputString(asJsonStr, "\"false\"", "false");
    output = nlohmann::json::parse(asJsonStr);
  }
  else{
    std::fstream fs;
    fs.open(configFilePath_, std::ios::in);

    if( not fs.is_open() ) {
      LogError << "\"" << configFilePath_ << "\": could not read file." << std::endl;
      throw std::runtime_error("file not readable.");
    }


    fs >> output;
  }

  return output;
}
