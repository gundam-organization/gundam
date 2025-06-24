//
// Created by Nadrino on 24/06/2025.
//

#include "FitSequencer.h"


void FitSequencer::configureImpl(){
  _config_.clearFields();
  _config_.defineFields({
    {"sequenceList"},
  });
}
void FitSequencer::initializeImpl(){

}
