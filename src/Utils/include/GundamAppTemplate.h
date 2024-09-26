//
// Created by Nadrino on 26/09/2024.
//

#ifndef GUNDAM_GUNDAMAPPTEMPLATE_H
#define GUNDAM_GUNDAMAPPTEMPLATE_H

#include <utility>

#include "ConfigUtils.h"

#include "CmdLineParser.h"
#include "Logger.h"


class GundamAppTemplate : public GenericToolbox::ConfigBaseClass {

public:
  GundamAppTemplate() = default;
  ~GundamAppTemplate() override = default;

  // core
  virtual void run(){}


protected:
  // to be called within configureImpl() override;
  virtual void defineCommandLineOptions(){
    LogInfo << _clp_.getDescription().str() << std::endl;

    LogInfo << "Usage: " << std::endl;
    LogInfo << _clp_.getConfigSummary() << std::endl << std::endl;
  }
  virtual void readCommandLineOptions(){
    _clp_.parseCmdLine(_argc_, _argv_);

    LogThrowIf(_clp_.isNoOptionTriggered(), "No option was provided.");

    LogInfo << "Provided arguments: " << std::endl;
    LogInfo << _clp_.getValueSummary() << std::endl << std::endl;
    LogInfo << _clp_.dumpConfigAsJsonStr() << std::endl;
  }

  int _argc_{0};
  char** _argv_{nullptr};
  std::string _name_{};
  CmdLineParser _clp_{};


};

#endif //GUNDAM_GUNDAMAPPTEMPLATE_H
