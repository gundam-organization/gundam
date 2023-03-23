//
// Created by Adrien Blanchet on 23/01/2023.
//

#include "GeneralSpline.h"
#include "CalculateGeneralSpline.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"


LoggerInit([]{
  Logger::setUserHeaderStr("[MonotonicSplineHandler]");
});

void GeneralSpline::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}

void GeneralSpline::buildSplineData(TGraph& graph_){
  // Copy the spline data into local storage.
  graph_.Sort();
  buildSplineData(TSpline3(Form("%p", &graph_), &graph_));
}
void GeneralSpline::buildSplineData(const TSpline3& sp_){
  LogThrowIf(not _splineData_.empty(), "Spline data already set.");

  _splineBounds_.first = sp_.GetXmin();
  _splineBounds_.second = sp_.GetXmax();

  _splineData_.resize(2 + sp_.GetNp()*3);
  _splineData_[0] = sp_.GetXmin();
  _splineData_[1] = ( ( sp_.GetXmax() - sp_.GetXmin() ) / ( sp_.GetNp()-1.0 ) );

  // Copy the spline data into local storage.
  for (int i = 0; i < sp_.GetNp(); ++i) {
    double x;
    double y;
    sp_.GetKnot(i,x,y);
    _splineData_[2 + 3*i + 0] = y;
    _splineData_[2 + 3*i + 1] = sp_.Derivative(x);
    _splineData_[2 + 3*i + 2] = x;
  }

//  LogThrow(GenericToolbox::parseVectorAsString(_splineData_));
}
double GeneralSpline::evaluateSpline(const DialInputBuffer& input_) const{
  double dialInput{input_.getBuffer()[0]};

  if( not _allowExtrapolation_ ){
    if     (input_.getBuffer()[0] <= _splineBounds_.first) { dialInput = _splineBounds_.first; }
    else if(input_.getBuffer()[0] >= _splineBounds_.second){ dialInput = _splineBounds_.second; }
  }

  return CalculateGeneralSpline( dialInput, -1E20, 1E20, _splineData_.data(), int(_splineData_.size()) );
}
