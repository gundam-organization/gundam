//
// Created by Nadrino on 24/06/2025.
//

#include "FitTask.h"

void FitTask::configureImpl(){
  _config_.clearFields();
  _config_.defineFields({
    {FieldFlag::MANDATORY, "name"},
    {"minimizerType"}
  });
}
void FitTask::initializeImpl(){

}
