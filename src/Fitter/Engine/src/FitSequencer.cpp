//
// Created by Nadrino on 24/06/2025.
//

#include "FitSequencer.h"


void FitSequencer::configureImpl(){
  _config_.clearFields();
  _config_.defineFields({
    {"sequenceList"},
  });
  _config_.checkConfiguration();

  _taskList_.clear();
  for( auto& taskEntry : _config_.loop("sequenceList") ) {
    FitTask task;
    task.configure(taskEntry);
    _taskList_.emplace_back(task);
  }
}
void FitSequencer::initializeImpl(){

}
