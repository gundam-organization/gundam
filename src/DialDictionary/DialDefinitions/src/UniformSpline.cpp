//
// Created by Adrien Blanchet on 23/01/2023.
//

/// @file UniformSpline.cpp
/// @brief Implements UniformSpline: packs knot values and slopes on a uniform
///        grid and evaluates them with the shared CalculateUniformSpline()
///        kernel.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in UniformSpline.h. The
/// comments here describe how each method does its job.

#include "UniformSpline.h"
#include "CalculateUniformSpline.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"

#include <limits>


/// @par Implementation
/// Stores the first position and the mean spacing, checks the spacing, then
/// interleaves the value and slope of each knot. Unlike CompactSpline and
/// MonotonicSpline, the spacing check always throws on failure, and its
/// tolerance (sqrt of the float epsilon) is much tighter than their 1%.
/// The slopes are stored in dy/dx units; the kernel multiplies them by the
/// spacing to get the Hermite tangents.
void UniformSpline::buildDial(const std::vector<DialUtils::DialPoint>& splinePointList_){
  LogThrowIf(not _splineData_.empty(), "Spline data already set.");

  _splineBounds_.min = splinePointList_.front().x;
  _splineBounds_.max = splinePointList_.back().x;

  _splineData_.resize(2 + splinePointList_.size()*2);
  _splineData_[0] = splinePointList_.front().x;
  _splineData_[1] = (splinePointList_.back().x-splinePointList_.front().x)/(splinePointList_.size()-1.0);

  /// Apply a very loose check that the point spacing is uniform to catch
  /// mistakes.  This only flags clear problems and isn't an accuracy
  /// guarrantee.  The tolerance is set based on "float" since the spline
  /// knots may have been saved or calculated using floats.
  const double tolerance{std::sqrt(std::numeric_limits<float>::epsilon())};
  for (int i=0; i<splinePointList_.size()-1; ++i) {
    double d = std::abs(splinePointList_[i].x - _splineData_[0] - i*_splineData_[1]); // distance of knot i from its expected position x0 + i*h

    if( (d/_splineData_[1])>tolerance ) {
      LogThrow("UniformSplines require uniformly spaced knots");
    }
  }

  // Copy the spline data into local storage.
  for (int i = 0; i < splinePointList_.size(); ++i) {
    double y = splinePointList_[i].y;
    double d = splinePointList_[i].slope; // slope dy/dx at knot i
    _splineData_[2 + 2*i + 0] = y;
    _splineData_[2 + 2*i + 1] = d;
  }

}

/// @par Implementation
/// Clamps the input to `_splineBounds_` unless extrapolation is allowed, then
/// calls CalculateUniformSpline() with the full data size as its `dim`
/// argument (the other uniform kernels take the knot count instead). The
/// kernel's output bounds are set to -1E20 and 1E20, so it does not clamp the
/// result. Outside the knot range the kernel still picks the first or last
/// interval, and the relative position in that interval falls below 0 or
/// above 1, which extends that interval's cubic.
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

/// @par Implementation
/// `_splineData_` holds `x0` and `h` first, then the a (value, slope) pair per knot, so
/// the knot count is read from its size and each position is `x0 + i h`.
std::vector<double> UniformSpline::getSourceKnotPositions() const {
  if( _splineData_.empty() ){ return {}; }
  std::vector<double> positions((_splineData_.size() - 2) / 2);
  for( size_t i = 0; i < positions.size(); ++i ){
    positions[i] = _splineData_[0] + i * _splineData_[1];
  }
  return positions;
}
