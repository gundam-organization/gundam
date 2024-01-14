//
// Created by Nadrino on 23/10/2022.
//

#ifndef GUNDAM_JSONBASECLASS_H
#define GUNDAM_JSONBASECLASS_H

#include "ConfigUtils.h"

#include "GenericToolbox.Utils.h"

#include "nlohmann/json.hpp"


class JsonBaseClass : public GenericToolbox::ConfigBaseClass<JsonType> {

public:
  JsonBaseClass() = default;

  void setConfig(const JsonType& config_) override{
    _config_ = config_;
    ConfigUtils::forwardConfig( _config_ );
  }

};

#endif //GUNDAM_JSONBASECLASS_H
