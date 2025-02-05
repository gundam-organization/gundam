//
// Created by Nadrino on 04/02/2025.
//

#ifndef ROOTMINIMIZERTASK_H
#define ROOTMINIMIZERTASK_H

#include "ConfigUtils.h"


class FitterEngine;

class FitterTask : JsonBaseClass {

public:
  FitterTask() = default;

  void run(FitterEngine *owner_);

protected:
  void configureImpl() override;

private:
  bool _isEnabled_{true};
  std::string _name_{};
  JsonType _config_{};

};



#endif //ROOTMINIMIZERTASK_H
