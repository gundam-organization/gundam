//
// Created by Adrien BLANCHET on 22/05/2021.
//

#ifndef XSLLHFITTER_JSONUTILS_H
#define XSLLHFITTER_JSONUTILS_H

#include "string"
#include "iostream"
#include "json.hpp"


namespace JsonUtils {

  nlohmann::json readConfigFile(const std::string& configFilePath_);

  template<class T> auto fetchValue(const nlohmann::json& jsonConfig_, const std::string& optionName_) -> T;
  template<class T> auto fetchValue(const nlohmann::json& jsonConfig_, const std::string& optionName_, const T& defaultValue_) -> T;

  class Loader{
  public:
    Loader();
  };

};


template<class T> auto JsonUtils::fetchValue(const nlohmann::json& jsonConfig_, const std::string& optionName_) -> T{
  auto jsonEntry = jsonConfig_.find(optionName_);
  if( jsonEntry == jsonConfig_.end() ){
    throw std::runtime_error("Could not find json entry: " + optionName_);
  }
  return jsonEntry->template get<T>();
}
template<class T> auto JsonUtils::fetchValue(const nlohmann::json& jsonConfig_, const std::string& optionName_, const T& defaultValue_) -> T{
  try{
    T value = JsonUtils::fetchValue<T>(jsonConfig_, optionName_);
    return value; // if nothing has gone wrong
  }
  catch (...){
    return defaultValue_;
  }
}

#endif //XSLLHFITTER_JSONUTILS_H
