//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialSpline.h"


double DialSpline::evalResponseImpl(const DialInputBuffer& input_) {
  return _spline_.Eval( input_.getBuffer()[0] );
}
