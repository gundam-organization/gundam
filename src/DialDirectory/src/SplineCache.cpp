//
// Created by Adrien Blanchet on 30/11/2022.
//

#include "SplineCache.h"


LoggerInit([]{
  Logger::setUserHeaderStr("[SplineCache]");
});

double SplineCache::evalResponseImpl(const DialInputBuffer& input_) {
  return this->calculateSplineResponse(input_);
}
