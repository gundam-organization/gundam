//
// Created by Nadrino on 22/05/2021.
//

#ifndef GUNDAM_JSONUTILS_H
#define GUNDAM_JSONUTILS_H

#include "nlohmann/json.hpp"

#include "string"
#include "vector"
#include "iostream"


namespace JsonUtils {

  nlohmann::json readConfigFile(const std::string& configFilePath_);
  nlohmann::json getForwardedConfig(const nlohmann::json& config_, const std::string& keyName_);
  void forwardConfig(nlohmann::json& config_, const std::string& className_ = "");
  void unfoldConfig(nlohmann::json& config_);

  std::vector<std::string> ls(const nlohmann::json& jsonConfig_);
  bool doKeyExist(const nlohmann::json& jsonConfig_, const std::string& keyName_);
  template<class T> auto fetchValue(const nlohmann::json& jsonConfig_, const std::string& keyName_) -> T;
  template<class T> auto fetchValue(const nlohmann::json& jsonConfig_, const std::vector<std::string>& keyNames_) -> T;
  template<class T> auto fetchValue(const nlohmann::json& jsonConfig_, const std::string& keyName_, const T& defaultValue_) -> T;
  template<class T> auto fetchValue(const nlohmann::json& jsonConfig_, const std::vector<std::string>& keyName_, const T& defaultValue_) -> T;
  template<class T> nlohmann::json fetchMatchingEntry(const nlohmann::json& jsonConfig_, const std::string& keyName_, const T& keyValue_);
  nlohmann::json fetchSubEntry(const nlohmann::json& jsonConfig_, const std::vector<std::string>& keyPath_);

  // template specialization when a string literal is passed:
  template<std::size_t N> auto fetchValue(const nlohmann::json& jsonConfig_, const std::string& keyName_, const char (&defaultValue_)[N]) -> std::string;
  template<std::size_t N> auto fetchValue(const nlohmann::json& jsonConfig_, const std::vector<std::string>& keyName_, const char (&defaultValue_)[N]) -> std::string;
  template<std::size_t N> nlohmann::json fetchMatchingEntry(const nlohmann::json& jsonConfig_, const std::string& keyName_, const char (&keyValue_)[N]);

};


#include "JsonUtils.impl.h"

#endif //GUNDAM_JSONUTILS_H
