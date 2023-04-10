//
// Created by Adrien Blanchet on 10/04/2023.
//

#include "Polynomial.h"


double Polynomial::evalResponse(const DialInputBuffer& input_) const {
  double result{0};
  double factor{1};
  for( auto coefficient : _coefficientList_ ) {
    result += coefficient * factor; // y += a_n * x^{n}
    factor *= input_.getBuffer()[0]; // x^{n} * x
  }
  return result;
}

void Polynomial::setCoefficientList(const std::vector<double> &coefficientList_) {
  _coefficientList_ = coefficientList_;
}
