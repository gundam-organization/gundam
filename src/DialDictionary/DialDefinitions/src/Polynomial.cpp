//
// Created by Adrien Blanchet on 10/04/2023.
//

#include "Polynomial.h"

double Polynomial::evalResponse(const DialInputBuffer& input_) const {
  double result{0};
  double factor{1};
  double dialInput{input_.getInputBuffer()[0]};

  if( not _allowExtrapolation_ ){
    if     ( dialInput <= _splineBounds_.min  ){ dialInput = _splineBounds_.min; }
    else if( dialInput >= _splineBounds_.max ){ dialInput = _splineBounds_.max; }
  }

  for( auto coefficient : _coefficientList_ ) {
    result += coefficient * factor; // y += a_n * x^{n}
    factor *= dialInput; // x^{n} * x
  }
  return result;
}
