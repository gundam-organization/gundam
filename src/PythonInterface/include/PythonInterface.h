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
  std::string _outRootFilePath_{};

  FitterEngine _fitter_;



public:
  PyGundam() = default;

  // configure stage
  void setConfig(const std::string& configPath_){ _configHandler_ = ConfigUtils::ConfigHandler(configPath_); }
  void addConfigOverride(const std::string& configPath_){ _configHandler_.override(configPath_); }

  // load
  void load(){
    _fitter_.setConfig( GenericToolbox::Json::fetchValue<JsonType>(_configHandler_.getConfig(), "fitterEngineConfig") );
    _fitter_.configure();

    _fitter_.getLikelihoodInterface().setForceAsimovData( true );
    _fitter_.initialize();
  }

  void fit(){ _fitter_.fit(); }

};


#endif //GUNDAM_PYTHON_INTERFACE_H
