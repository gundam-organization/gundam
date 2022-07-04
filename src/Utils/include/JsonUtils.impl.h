//
// Created by Nadrino on 22/05/2021.
//

#ifndef GUNDAM_JSONUTILS_IMPL_H
#define GUNDAM_JSONUTILS_IMPL_H

#include "JsonUtils.h"

#include "GenericToolbox.h"

#include "nlohmann/json.hpp"

#include "string"
#include "iostream"

namespace JsonUtils {

  template<class T> auto fetchValue(const nlohmann::json& jsonConfig_, const std::string& keyName_) -> T{
    auto jsonEntry = jsonConfig_.find(keyName_);
    if( jsonEntry == jsonConfig_.end() ){
      throw std::runtime_error("Could not find json entry: " + keyName_ + ":\n" + jsonConfig_.dump());
    }
    return jsonEntry->template get<T>();
  }
  template<class T> auto fetchValue(const nlohmann::json& jsonConfig_, const std::vector<std::string>& keyNames_) -> T{
    for( auto& keyName : keyNames_){
      if( JsonUtils::doKeyExist(jsonConfig_, keyName) ){
        return JsonUtils::fetchValue<T>(jsonConfig_, keyName);
      }
    }
    throw std::runtime_error("Could not find any json entry: " + GenericToolbox::parseVectorAsString(keyNames_) + ":\n" + jsonConfig_.dump());
  }
  template<class T> auto fetchValue(const nlohmann::json& jsonConfig_, const std::string& keyName_, const T& defaultValue_) -> T{
    try{
      T value = JsonUtils::fetchValue<T>(jsonConfig_, keyName_);
      return value; // if nothing has gone wrong
    }
    catch (...){
      return defaultValue_;
    }
  }
  template<class T> auto fetchValue(const nlohmann::json& jsonConfig_, const std::vector<std::string>& keyName_, const T& defaultValue_) -> T{
    for( auto& keyName : keyName_ ){
      try{
        T value = JsonUtils::fetchValue<T>(jsonConfig_, keyName);
        return value; // if nothing has gone wrong
      }
      catch (...){
      }
    }
    return defaultValue_;
  }
  template<class T> nlohmann::json fetchMatchingEntry(const nlohmann::json& jsonConfig_, const std::string& keyName_, const T& keyValue_){

    if( not jsonConfig_.is_array() ){
      std::cout << "key: " << keyName_ << std::endl;
      std::cout << "value: " << keyValue_ << std::endl;
      std::cout << "dump: " << jsonConfig_.dump() << std::endl;
      throw std::runtime_error("JsonUtils::fetchMatchingEntry: jsonConfig_ is not an array.");
    }

    for( const auto& jsonEntry : jsonConfig_ ){
      try{
        if(JsonUtils::fetchValue<T>(jsonEntry, keyName_) == keyValue_ ){
          return jsonEntry;
        }
      }
      catch (...){
        // key not present, skip
      }

    }

    return nlohmann::json(); // .empty()
  }

  // specialization
  template<std::size_t N> auto fetchValue(const nlohmann::json& jsonConfig_, const std::string& keyName_, const char (&defaultValue_)[N]) -> std::string{
    return fetchValue(jsonConfig_, keyName_, std::string(defaultValue_));
  }
  template<std::size_t N> auto fetchValue(const nlohmann::json& jsonConfig_, const std::vector<std::string>& keyName_, const char (&defaultValue_)[N]) -> std::string{
    return fetchValue(jsonConfig_, keyName_, std::string(defaultValue_));
  }
  template<std::size_t N> nlohmann::json fetchMatchingEntry(const nlohmann::json& jsonConfig_, const std::string& keyName_, const char (&keyValue_)[N]){
    return fetchMatchingEntry(jsonConfig_, keyName_, std::string(keyValue_));
  }

  // gundam specific
  std::string buildFormula(const nlohmann::json& jsonConfig_, const std::string& keyName_, const std::string& joinStr_);
  std::string buildFormula(const nlohmann::json& jsonConfig_, const std::string& keyName_, const std::string& joinStr_, const std::string& defaultFormula_);

};


#endif //GUNDAM_JSONUTILS_IMPL_H
