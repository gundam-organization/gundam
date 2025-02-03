//
// Created by Adrien Blanchet on 05/12/2023.
//

#ifndef GUNDAM_PYTHON_INTERFACE_H
#define GUNDAM_PYTHON_INTERFACE_H


#include "FitterEngine.h"
#include "ConfigUtils.h"
#include "GundamApp.h"

#include <string>


class PyGundam{

  GundamApp app{"PyGundam"};
  ConfigUtils::ConfigHandler _configHandler_;
  FitterEngine _fitter_;

public:
  PyGundam() = default;

  // configure stage
  void setOutputRootFilePath(const std::string& outRootFilePath_){ app.openOutputFile(outRootFilePath_); }
  void setConfig(const std::string& configPath_){ _configHandler_ = ConfigUtils::ConfigHandler(configPath_); }
  void addConfigOverride(const std::string& configPath_){ _configHandler_.override(configPath_); }

  // load
  void load();

  void minimize(){ _fitter_.getMinimizer().minimize(); }

  double getVal(){
    return _fitter_.getLikelihoodInterface().evalLikelihood();
  }

  // getters
  FitterEngine& getFitterEngine(){ return _fitter_; }

};


#endif //GUNDAM_PYTHON_INTERFACE_H
