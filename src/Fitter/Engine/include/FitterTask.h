//
// Created by Nadrino on 04/02/2025.
//

#ifndef ROOTMINIMIZERTASK_H
#define ROOTMINIMIZERTASK_H

#include "ConfigUtils.h"


class FitterEngine;

class FitterTask {

public:
  FitterTask() = default;

  void setConfig(const JsonType& config_){ _config_ = config_; }

  void run(FitterEngine *owner_);

private:
  JsonType _config_{};

};



#endif //ROOTMINIMIZERTASK_H
