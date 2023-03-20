//
// Created by Adrien Blanchet on 22/01/2023.
//

#include "CompactSplineHandler.h"
#include "CalculateCompactSpline.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"


LoggerInit([]{
  Logger::setUserHeaderStr("[CompactSplineHandler]");
});

void CompactSplineHandler::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}

void CompactSplineHandler::buildSplineData(TGraph& graph_){
  LogThrowIf(not _splineData_.empty(), "Spline data already set.");

  // Copy the spline data into local storage.
  graph_.Sort();

  LogThrowIf(
      not GenericToolbox::hasUniformlySpacedKnots(&graph_),
      "Can't use compact spline with a input that doesn't have uniformly spaced knots"
  );

  _splineBounds_.first = graph_.GetX()[0];
  _splineBounds_.second = graph_.GetX()[graph_.GetN()-1];

  _splineData_.resize(2 + graph_.GetN());
  _splineData_[0] = graph_.GetX()[0];
  _splineData_[1] = (graph_.GetX()[graph_.GetN()-1] - graph_.GetX()[0])/(graph_.GetN()-1.0);

  memcpy(&_splineData_[2], graph_.GetY(), graph_.GetN() * sizeof(double));
}
double CompactSplineHandler::evaluateSpline(const DialInputBuffer& input_) const{
  double dialInput{input_.getBuffer()[0]};

  if( not _allowExtrapolation_ ){
    if     (input_.getBuffer()[0] <= _splineBounds_.first) { dialInput = _splineBounds_.first; }
    else if(input_.getBuffer()[0] >= _splineBounds_.second){ dialInput = _splineBounds_.second; }
  }

  return CalculateCompactSpline( dialInput, -1E20, 1E20, _splineData_.data(), int(_splineData_.size()) );
}
