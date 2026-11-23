//
// Created by Adrien Blanchet on 10/04/2023.
//

/// @file Polynomial.cpp
/// @brief Implements Polynomial::evalResponse().
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in Polynomial.h. The
/// comments here describe how each method does its job.

#include "Polynomial.h"

/// @par Implementation
/// Clamps the input first (when extrapolation is off), then adds the terms
/// in increasing order, keeping a running power of x instead of calling
/// std::pow().
double Polynomial::evalResponse(const DialInputBuffer& input_) const {
  double result{0};
  double factor{1}; // x^n for the current coefficient
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
