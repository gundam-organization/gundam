//
// Created by Nadrino on 17/06/2021.
//

#ifndef GUNDAM_YAMLUTILS_H
#define GUNDAM_YAMLUTILS_H

#include "nlohmann/json.hpp"
#include "yaml-cpp/yaml.h"

#include "string"
#include "iostream"


namespace YamlUtils {

  YAML::Node readConfigFile(const std::string& configFilePath_);
  nlohmann::json toJson(const YAML::Node& yamlConfig_);

  template<class T> auto fetchValue(const YAML::Node& yamlConfig_, const std::string& keyName_) -> T;
  template<class T> auto fetchValue(const YAML::Node& yamlConfig_, const std::string& keyName_, const T& defaultValue_) -> T;
  template<class T> YAML::Node fetchMatchingEntry(const YAML::Node& yamlConfig_, const std::string& keyName_, const T& keyValue_);

  // template specialization when a string literal is passed:
  template<std::size_t N> auto fetchValue(const YAML::Node& yamlConfig_, const std::string& keyName_, const char (&defaultValue_)[N]) -> std::string;
  template<std::size_t N> YAML::Node fetchMatchingEntry(const YAML::Node& yamlConfig_, const std::string& keyName_, const char (&keyValue_)[N]);

};

#include "YamlUtils.impl.h"

#endif //GUNDAM_YAMLUTILS_H
