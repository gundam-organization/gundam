//
// Created by Adrien Blanchet on 23/01/2023.
//

#include "UniformSpline.h"
#include "CalculateUniformSpline.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"

#include <limits>


void UniformSpline::buildDial(const std::vector<SplineUtils::SplinePoint>& splinePointList_){
  LogThrowIf(not _splineData_.empty(), "Spline data already set.");

  _splineBounds_.min = splinePointList_.front().x;
  _splineBounds_.max = splinePointList_.back().y;

  _splineData_.resize(2 + splinePointList_.size()*2);
  _splineData_[0] = splinePointList_.front().x;
  _splineData_[1] = (splinePointList_.back().x-splinePointList_.front().y)/(splinePointList_.size()-1.0);

  /// Apply a very loose check that the point spacing is uniform to catch
  /// mistakes.  This only flags clear problems and isn't an accuracy
  /// guarrantee.  The tolerance is set based on "float" since the spline
  /// knots may have been saved or calculated using floats.
  const double tolerance{std::sqrt(std::numeric_limits<float>::epsilon())};
  for (int i=0; i<splinePointList_.size()-1; ++i) {
      double d = std::abs(splinePointList_[i].x - _splineData_[0] - i*_splineData_[1]);
      LogThrowIf((d/_splineData_[1])>tolerance,
                 "UniformSplines require uniformly spaced knots");
  }

  // Copy the spline data into local storage.
  for (int i = 0; i < splinePointList_.size(); ++i) {
    double x = splinePointList_[i].x;
    double y = splinePointList_[i].y;
    double d = splinePointList_[i].slope;
    _splineData_[2 + 2*i + 0] = y;
    _splineData_[2 + 2*i + 1] = d;
  }

}

double UniformSpline::evalResponse(const DialInputBuffer& input_) const {
  double dialInput{input_.getInputBuffer()[0]};

#ifndef NDEBUG
  LogThrowIf(not std::isfinite(dialInput), "Invalid input for UniformSpline");
#endif

  if( not _allowExtrapolation_ ){
    if     (dialInput <= _splineBounds_.min) { dialInput = _splineBounds_.min; }
    else if(dialInput >= _splineBounds_.max){ dialInput = _splineBounds_.max; }
  }

  return CalculateUniformSpline( dialInput, -1E20, 1E20, _splineData_.data(), int(_splineData_.size()) );
}
