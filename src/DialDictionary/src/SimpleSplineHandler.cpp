//
// Created by Adrien Blanchet on 24/01/2023.
//

#include "SimpleSplineHandler.h"
#include "CalculateGeneralSpline.h"
#include "CalculateMonotonicSpline.h"

#include "Logger.h"
#include "GenericToolbox.Root.h"


LoggerInit([]{
  Logger::setUserHeaderStr("[MonotonicSplineHandler]");
});

void SimpleSplineHandler::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}

void SimpleSplineHandler::buildSplineData(TGraph& graph_){
  LogThrowIf(not _splineData_.empty(), "Spline data already set.");

  // Copy the spline data into local storage.
  graph_.Sort();

  if( GenericToolbox::hasUniformlySpacedKnots(&graph_) ){
    _isMonotonic_ = true;

    _splineBounds_.first = graph_.GetX()[0];
    _splineBounds_.second = graph_.GetX()[graph_.GetN()-1];

    _splineData_.resize(2 + graph_.GetN());
    _splineData_[0] = graph_.GetX()[0];
    _splineData_[1] = (graph_.GetX()[graph_.GetN()-1] - graph_.GetX()[0])/(graph_.GetN()-1.0);

    memcpy(&_splineData_[2], graph_.GetY(), graph_.GetN() * sizeof(double));
  }
  else{
    TSpline3 sp(Form("%p", &graph_), &graph_);
    _splineBounds_.first = sp.GetXmin();
    _splineBounds_.second = sp.GetXmax();

    _splineData_.resize(2 + sp.GetNp() * 3);
    _splineData_[0] = sp.GetXmin();
    _splineData_[1] = ((sp.GetXmax() - sp.GetXmin() ) / (sp.GetNp() - 1.0 ) );

    // Copy the spline data into local storage.
    for (int i = 0; i < sp.GetNp(); ++i) {
      double x;
      double y;
      sp.GetKnot(i, x, y);
      _splineData_[2 + 3*i + 0] = y;
      _splineData_[2 + 3*i + 1] = sp.Derivative(x);
      _splineData_[2 + 3*i + 2] = x;
    }
  }

}
void SimpleSplineHandler::buildSplineData(const TSpline3& sp_){
  LogThrow("NOT IMPLEMENTED");


  LogThrowIf(not _splineData_.empty(), "Spline data already set.");


}
double SimpleSplineHandler::evaluateSpline(const DialInputBuffer& input_) const{
  double dialInput{input_.getBuffer()[0]};

  if( not _allowExtrapolation_ ){
    if     (input_.getBuffer()[0] <= _splineBounds_.first) { dialInput = _splineBounds_.first; }
    else if(input_.getBuffer()[0] >= _splineBounds_.second){ dialInput = _splineBounds_.second; }
  }

  if( _isMonotonic_ ){
    return CalculateMonotonicSpline( dialInput, -1E20, 1E20, _splineData_.data(), int(_splineData_.size()) );
  }
  return CalculateGeneralSpline( dialInput, -1E20, 1E20, _splineData_.data(), int(_splineData_.size()) );
}

