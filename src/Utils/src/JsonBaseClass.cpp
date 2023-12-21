//
// Created by Adrien BLANCHET on 23/10/2022.
//

#include "JsonBaseClass.h"

#include "ConfigUtils.h"


void JsonBaseClass::setConfig(const JsonType& config_){
  _config_ = config_;
  ConfigUtils::forwardConfig(_config_);
}
