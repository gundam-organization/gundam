//
// Created by Adrien BLANCHET on 23/10/2022.
//

#include "JsonBaseClass.h"

#include "JsonUtils.h"

void JsonBaseClass::setConfig(const nlohmann::json& config_){
  _config_ = config_;
  JsonUtils::forwardConfig(_config_);
}
