//
// Created by Adrien BLANCHET on 11/06/2021.
//

#ifndef XSLLHFITTER_PARAMETERPROPAGATOR_H
#define XSLLHFITTER_PARAMETERPROPAGATOR_H

#include "vector"

#include "FitParameterSet.h"
#include "AnaSample.hh"

class ParameterPropagator {

public:
  ParameterPropagator();
  virtual ~ParameterPropagator();

  void propagateParametersOnSample(const AnaSample& sample_);

private:
  std::vector<FitParameterSet> _parameterSetsList_;

};


#endif //XSLLHFITTER_PARAMETERPROPAGATOR_H
