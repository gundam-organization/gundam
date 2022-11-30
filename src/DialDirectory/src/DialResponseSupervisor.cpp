//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialResponseSupervisor.h"


void DialResponseSupervisor::process(double& output_) const{
  // apply cap?

  for( auto& transformFunction : _functionsList_ ){
    transformFunction(output_);
  }

}