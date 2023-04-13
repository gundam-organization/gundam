//
// Created by Adrien BLANCHET on 23/10/2022.
//

#ifndef GUNDAM_JSONBASECLASS_H
#define GUNDAM_JSONBASECLASS_H

#include "GenericToolbox.ConfigBaseClass.h"

#include "nlohmann/json.hpp"

class JsonBaseClass : public GenericToolbox::ConfigBaseClass<nlohmann::json> {

public:
  JsonBaseClass() = default;

  void setConfig(const nlohmann::json& config_) override;


};

#endif //GUNDAM_JSONBASECLASS_H
