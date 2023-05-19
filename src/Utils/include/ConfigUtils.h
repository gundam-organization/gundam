//
// Created by Adrien Blanchet on 28/02/2023.
//

#ifndef GUNDAM_CONFIGUTILS_H
#define GUNDAM_CONFIGUTILS_H

#include "nlohmann/json.hpp"
#include "yaml-cpp/yaml.h"

#include "string"


namespace ConfigUtils {

  // could be YAML or JSON
  nlohmann::json readConfigFile(const std::string& configFilePath_);
  nlohmann::json convertYamlToJson(const std::string& configFilePath_);
  nlohmann::json convertYamlToJson(const YAML::Node& yamlConfig_);

  // make sure both YAML and JSON are supported
  nlohmann::json getForwardedConfig(const nlohmann::json& config_);
  nlohmann::json getForwardedConfig(const nlohmann::json& config_, const std::string& keyName_);
  void forwardConfig(nlohmann::json& config_, const std::string& className_ = "");
  void unfoldConfig(nlohmann::json& config_);

}

#endif //GUNDAM_CONFIGUTILS_H
