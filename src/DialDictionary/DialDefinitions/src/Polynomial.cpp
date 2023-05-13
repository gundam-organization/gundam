//
// Created by Adrien Blanchet on 10/04/2023.
//

#include "Polynomial.h"

double Polynomial::evalResponse(const DialInputBuffer& input_) const {
  double result{0};
  double factor{1};
  double dialInput{input_.getBuffer()[0]};

  if( not _allowExtrapolation_ ){
    if     (input_.getBuffer()[0] <= _splineBounds_.first) { dialInput = _splineBounds_.first; }
    else if(input_.getBuffer()[0] >= _splineBounds_.second){ dialInput = _splineBounds_.second; }
  }

  for( auto coefficient : _coefficientList_ ) {
    result += coefficient * factor; // y += a_n * x^{n}
    factor *= dialInput; // x^{n} * x
  }
  return result;
}

void Polynomial::buildDial(const std::vector<double>& coefficientList_,
                       const std::vector<double>&,
                       const std::vector<double>&,
                       const std::string& option_) {
  setCoefficientList( coefficientList_ );
}
