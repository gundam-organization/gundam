//
// Created by Nadrino on 17/06/2021.
//

#ifndef GUNDAM_YAMLUTILS_IMPL_H
#define GUNDAM_YAMLUTILS_IMPL_H

#include "string"
#include "iostream"
#include "exception"

#include "yaml-cpp/yaml.h"

#include "Logger.h"
#include "GenericToolbox.h"

#include "YamlUtils.h"

// https://github.com/jbeder/yaml-cpp/wiki/Tutorial

namespace YamlUtils {

  template<class T> auto fetchValue(const YAML::Node& yamlConfig_, const std::string& keyName_) -> T{

    if( keyName_.empty() ){
      throw std::runtime_error("Could not fetch in YAML node: keyName_.empty()");
    }
    else if( yamlConfig_.IsNull() ){
      throw std::runtime_error("Could not fetch in YAML node: IsNull()");
    }

    if( not yamlConfig_[keyName_] ){
      throw std::runtime_error(keyName_ + " does not exist");
    }

    return yamlConfig_[keyName_].template as<T>();
  }
  template<class T> auto fetchValue(const YAML::Node& yamlConfig_, const std::string& keyName_, const T& defaultValue_) -> T{
    try{
      T value = fetchValue<T>(yamlConfig_, keyName_);
      return value; // if nothing has gone wrong
    }
    catch (...){
      return defaultValue_;
    }
  }
  template<class T> YAML::Node fetchMatchingEntry(const YAML::Node& yamlConfig_, const std::string& keyName_, const T& keyValue_){

    if( not yamlConfig_.IsSequence() ){
      throw std::runtime_error("Could not fetchMatchingEntry in YAML node: IsSequence() == false");
    }

    for( const auto& yamlEntry : yamlConfig_ ){
      if(yamlEntry[keyName_] and yamlEntry[keyName_].template as<T>() == keyValue_ ){
        return YAML::Node(yamlEntry);
      }
    }

    return YAML::Node(); // .empty()
  }

  template<std::size_t N> auto fetchValue(const YAML::Node& yamlConfig_, const std::string& keyName_, const char (&defaultValue_)[N]) -> std::string{
    return fetchValue(yamlConfig_, keyName_, std::string(defaultValue_));
  }
  template<std::size_t N> YAML::Node fetchMatchingEntry(const YAML::Node& yamlConfig_, const std::string& keyName_, const char (&keyValue_)[N]){
    return fetchMatchingEntry(yamlConfig_, keyName_, std::string(keyValue_));
  }

};

#endif //GUNDAM_YAMLUTILS_IMPL_H
