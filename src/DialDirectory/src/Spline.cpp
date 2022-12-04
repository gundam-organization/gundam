//
// Created by Adrien Blanchet on 30/11/2022.
//

#include "Spline.h"

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[Spline]");
});

double Spline::evalResponseImpl(const DialInputBuffer& input_) {
  return this->calculateSplineResponse(input_);
}
