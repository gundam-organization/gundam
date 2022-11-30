//
// Created by Adrien Blanchet on 30/11/2022.
//

#include "SplineCache.h"

double SplineCache::evalResponseImpl(const DialInputBuffer& input_) {
  return _spline_.Eval( input_.getBuffer()[0] );
}
