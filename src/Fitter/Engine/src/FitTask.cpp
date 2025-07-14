//
// Created by Nadrino on 24/06/2025.
//

#include "FitTask.h"

void FitTask::configureImpl(){
  _config_.clearFields();
  _config_.defineFields({
    {FieldFlag::MANDATORY, "name"},
    {"minimizerType"},
    {"actions"},
    {"outputFolder"},
  });
  _config_.checkConfiguration();

  _config_.fillValue(_name_, "name");
  _config_.fillValue(_isEnabled_, "isEnabled");
  if( not _isEnabled_ ){ return; } // don't even read the rest of the config

  _config_.fillValue(_actionList_, "actions");
  _config_.fillValue(_outputFolder_, "outputFolder");
}
void FitTask::initializeImpl(){

}

void FitTask::run(){

}
