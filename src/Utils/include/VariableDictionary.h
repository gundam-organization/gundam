//
// Created by Adrien BLANCHET on 10/11/2022.
//

#ifndef GUNDAM_VARIABLEDICTIONARY_H
#define GUNDAM_VARIABLEDICTIONARY_H

#include "JsonBaseClass.h"

#include "nlohmann/json.hpp"

#include <string>

struct VariableDictEntry{
  VariableDictEntry() = default;
  VariableDictEntry(const nlohmann::json& config_);

  void readConfig(const nlohmann::json& config_);

  std::string name{};
  std::string displayName{};
  std::string unit{};
  std::string description{};
};

class VariableDictionary {

public:
  void fillDictionary(const nlohmann::json& config_, bool overrideIfDefined_ = true);
  bool isVariableDefined(const std::string& variableName_) const;
  const VariableDictEntry& getEntry(const std::string& variableName_) const;
  VariableDictEntry& getEntry(const std::string& variableName_);

private:
  std::vector<VariableDictEntry> dictionary;
};

#endif //GUNDAM_VARIABLEDICTIONARY_H
