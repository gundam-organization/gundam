//
// Created by Nadrino on 23/01/2023.
//

#include "GeneralSpline.h"
#include "CalculateGeneralSpline.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"


LoggerInit([]{
  Logger::setUserHeaderStr("[GeneralSpline]");
});

void GeneralSpline::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}

bool GeneralSpline::getAllowExtrapolation() const {
  return _allowExtrapolation_;
}

void GeneralSpline::buildDial(const TGraph& graph_, const std::string& option_){
  // Copy the spline data into local storage.
  TGraph grf(graph_);
  grf.Sort();
  buildDial(TSpline3(Form("%p", &graph_), &graph_), option_);
}

void GeneralSpline::buildDial(const TSpline3& sp_, const std::string& option_){
  std::vector<double> xPoints;
  std::vector<double> yPoints;
  std::vector<double> deriv;

  // Copy the spline data into local storage.
  for (int i = 0; i < sp_.GetNp(); ++i) {
    double x;
    double y;
    sp_.GetKnot(i,x,y);
    double d = sp_.Derivative(x);
    xPoints.push_back(x);
    yPoints.push_back(y);
    deriv.push_back(d);
  }
  buildDial(xPoints, yPoints, deriv);
}

void GeneralSpline::buildDial(const std::vector<double>& xPoints,
                              const std::vector<double>& yPoints,
                              const std::vector<double>& deriv,
                              const std::string& option_){
  LogThrowIf(not _splineData_.empty(), "Spline data already set.");

  _splineBounds_.first = xPoints.front();
  _splineBounds_.second = xPoints.back();

  _splineData_.resize(2 + xPoints.size()*3);
  _splineData_[0] = xPoints.front();
  _splineData_[1] = (xPoints.back()-xPoints.front())/(xPoints.size()-1.0);

  // Copy the spline data into local storage.
  for (int i = 0; i < xPoints.size(); ++i) {
    double x = xPoints[i];
    double y = yPoints[i];
    double d = deriv[i];
    _splineData_[2 + 3*i + 0] = y;
    _splineData_[2 + 3*i + 1] = d;
    _splineData_[2 + 3*i + 2] = x;
  }

//  LogThrow(GenericToolbox::toString(_splineData_));
}

double GeneralSpline::evalResponse(const DialInputBuffer& input_) const {
  double dialInput{input_.getBuffer()[0]};

#ifndef NDEBUG
  LogThrowIf(not std::isfinite(dialInput), "Invalid input for GeneralSpline");
#endif

  if( not _allowExtrapolation_ ){
    if     (dialInput <= _splineBounds_.first) { dialInput = _splineBounds_.first; }
    else if(dialInput >= _splineBounds_.second){ dialInput = _splineBounds_.second; }
  }

  return CalculateGeneralSpline( dialInput, -1E20, 1E20, _splineData_.data(), int(_splineData_.size()) );
}
