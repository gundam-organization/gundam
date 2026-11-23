//
// Created by Adrien Blanchet on 23/01/2023.
//

/// @file GeneralSpline.cpp
/// @brief Implements GeneralSpline: packs knot positions, values and slopes
///        and evaluates them with the shared CalculateGeneralSpline() kernel.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in GeneralSpline.h. The
/// comments here describe how each method does its job.

#include "GeneralSpline.h"
#include "CalculateGeneralSpline.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"


/// @par Implementation
/// Fills the two header values (first position and mean spacing) only so the
/// layout starts like the other spline dials, then copies one
/// (value, slope, position) triple per knot. No spacing or ordering check is
/// made: the points are expected to be sorted already.
void GeneralSpline::buildDial(const std::vector<DialUtils::DialPoint>& splinePointList_){
  LogThrowIf(not _splineData_.empty(), "Spline data already set.");

  _splineBounds_.min = splinePointList_.front().x;
  _splineBounds_.max = splinePointList_.back().x;

  _splineData_.resize(2 + splinePointList_.size()*3);
  _splineData_[0] = splinePointList_.front().x;
  _splineData_[1] = (splinePointList_.back().x-splinePointList_.front().x)/(splinePointList_.size()-1.0);

  // Copy the spline data into local storage.
  for (size_t iPt = 0; iPt < splinePointList_.size(); ++iPt) {
    double x = splinePointList_[iPt].x;
    double y = splinePointList_[iPt].y;
    double d = splinePointList_[iPt].slope; // slope dy/dx at knot iPt
    _splineData_[2 + 3*iPt + 0] = y;
    _splineData_[2 + 3*iPt + 1] = d;
    _splineData_[2 + 3*iPt + 2] = x;
  }
}

/// @par Implementation
/// Clamps the input to `_splineBounds_` unless extrapolation is allowed, then
/// calls CalculateGeneralSpline() with the full data size as its `dim`
/// argument. The kernel's output bounds are set to -1E20 and 1E20, so it does
/// not clamp the result. The kernel finds the interval with a fixed
/// five-step binary search written to run well on the GPU; outside the knot
/// range it stays on the first or last interval and so extends that
/// interval's cubic. Unlike the other spline dials, there is no debug-build
/// check for a non-finite input.
double GeneralSpline::evalResponse(const DialInputBuffer& input_) const {
  double dialInput{input_.getInputBuffer()[0]};

  if( not _allowExtrapolation_ ){
    if     (dialInput <= _splineBounds_.min) { dialInput = _splineBounds_.min; }
    else if(dialInput >= _splineBounds_.max){ dialInput = _splineBounds_.max; }
  }

  return CalculateGeneralSpline( dialInput, -1E20, 1E20, _splineData_.data(), int(_splineData_.size()) );
}

/// @par Implementation
/// Skips the two header values and the value and slope of the first knot,
/// then reads every third entry: the position of each (value, slope,
/// position) triple.
std::vector<double> GeneralSpline::getSourceKnotPositions() const {
  std::vector<double> positions;
  positions.reserve(_splineData_.size() / 3);
  for( size_t i = 4; i < _splineData_.size(); i += 3 ){
    positions.emplace_back(_splineData_[i]);
  }
  return positions;
}
