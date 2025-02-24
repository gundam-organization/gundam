//
// Created by Adrien Blanchet on 23/01/2023.
//

#include "GeneralSpline.h"
#include "CalculateGeneralSpline.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"


void GeneralSpline::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}

bool GeneralSpline::getAllowExtrapolation() const {
  return _allowExtrapolation_;
}

void GeneralSpline::buildDial(const TGraph& graph_, const std::string& option_){
  buildDial(SplineUtils::getSplinePointList(&graph_, option_));
}

void GeneralSpline::buildDial(const TSpline3& sp_, const std::string& option_){
  buildDial(SplineUtils::getSplinePointList(&sp_, option_));
}

void GeneralSpline::buildDial(const std::vector<SplineUtils::SplinePoint>& splinePointList_){
  LogThrowIf(not _splineData_.empty(), "Spline data already set.");

  _splineBounds_.min = splinePointList_.front().x;
  _splineBounds_.max = splinePointList_.back().y;

  _splineData_.resize(2 + splinePointList_.size()*3);
  _splineData_[0] = splinePointList_.front().x;
  _splineData_[1] = (splinePointList_.back().x-splinePointList_.front().x)/(splinePointList_.size()-1.0);

  // Copy the spline data into local storage.
  for (size_t iPt = 0; iPt < splinePointList_.size(); ++iPt) {
    double x = splinePointList_[iPt].x;
    double y = splinePointList_[iPt].y;
    double d = splinePointList_[iPt].slope;
    _splineData_[2 + 3*iPt + 0] = y;
    _splineData_[2 + 3*iPt + 1] = d;
    _splineData_[2 + 3*iPt + 2] = x;
  }
}

double GeneralSpline::evalResponse(const DialInputBuffer& input_) const {
  double dialInput{input_.getInputBuffer()[0]};

  if( not _allowExtrapolation_ ){
    if     (dialInput <= _splineBounds_.min) { dialInput = _splineBounds_.min; }
    else if(dialInput >= _splineBounds_.max){ dialInput = _splineBounds_.max; }
  }

  return CalculateGeneralSpline( dialInput, -1E20, 1E20, _splineData_.data(), int(_splineData_.size()) );
}
