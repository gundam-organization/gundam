//
// Created by Adrien Blanchet on 30/11/2022.
//

#include "Spline.h"

double Spline::evalResponseImpl(const DialInputBuffer& input_) {
  return _spline_.Eval( input_.getBuffer()[0] );
}
