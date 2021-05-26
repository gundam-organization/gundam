//
// Created by Adrien BLANCHET on 21/05/2021.
//

#ifndef XSLLHFITTER_DIAL_H
#define XSLLHFITTER_DIAL_H

#include "GenericToolbox.h"

ENUM_EXPANDER(
  PropagationMethod, -1,
  Invalid,
  ReWeight,
  ResponseMatrix
);


class Dial {

public:
  Dial();
  virtual ~Dial();

  void reset();

  double evalDial(const double& parameterValue_);

private:
  double _lastEvalDial_;
  double _lastEvalParameter_;

  PropagationMethod _propagationMethod_{Invalid};

};


#endif //XSLLHFITTER_DIAL_H
