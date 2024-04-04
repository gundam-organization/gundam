//
// Created by Adrien BLANCHET on 23/10/2022.
//

#ifndef GUNDAM_JSON_BASE_CLASS_H
#define GUNDAM_JSON_BASE_CLASS_H

#include "ConfigUtils.h"

#include "GenericToolbox.Utils.h"

#include "nlohmann/json.hpp"


class JsonBaseClass : public GenericToolbox::ConfigBaseClass<JsonType> {

public:
  JsonBaseClass() = default;

  void setConfig(const JsonType& config_) override { _config_ = config_; }


};

#endif //GUNDAM_JSON_BASE_CLASS_H
