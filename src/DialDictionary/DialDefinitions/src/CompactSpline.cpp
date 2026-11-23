//
// Created by Adrien Blanchet on 22/01/2023.
//

/// @file CompactSpline.cpp
/// @brief Implements CompactSpline: packs uniformly spaced knot values and
///        evaluates them with the shared CalculateCompactSpline() kernel.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in CompactSpline.h. The
/// comments here describe how each method does its job.

#include "CompactSpline.h"
#include "CalculateCompactSpline.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"

/// @par Implementation
/// Stores the first position and the mean spacing, then checks that every
/// knot lies within 1% of a spacing of its expected position `x0 + i h`.
/// This is a last safety net: DialCollection::makeSplineDial() already tests
/// the spacing with DialUtils::isUniform(), but with its default tolerance it
/// only logs an error and goes on. The knot values are copied last.
void CompactSpline::buildDial(const std::vector<DialUtils::DialPoint>& splinePointList_){
  LogThrowIf(not _splineData_.empty(), "Spline data already set.");

  _splineBounds_.min = splinePointList_.front().x;
  _splineBounds_.max = splinePointList_.back().x;

  _splineData_.resize(2 + splinePointList_.size());
  _splineData_[0] = splinePointList_.front().x;
  _splineData_[1] = (splinePointList_.back().x - splinePointList_.front().x)/(splinePointList_.size()-1.0);

  // Non-uniform input points should be caught before the CompactSpline is
  // built, but apply a final sanity check to make sure the point spacing is
  // uniform.  This only flags very clear mistakes and isn't an accuracy
  // guarrantee.  The tolerance is set as a fraction of the point spacing.
  const double tolerance{1E-2};
  bool validInputs = true;
  // First check that the points have a reasonable separation.
  if (_splineData_[1] < 1E-6) validInputs = false;
  // Make sure the points are uniformly spaced.
  for (int i=0; i<splinePointList_.size()-1; ++i) {
      double d = std::abs(splinePointList_[i].x - _splineData_[0] - i*_splineData_[1]); // distance of knot i from its expected position x0 + i*h
      if ((d/_splineData_[1])>tolerance) validInputs = false;
  }
  // Make lots of output if there is a problem!  This hopefully gives a clue
  // which spline is causing trouble.
  if (not validInputs) {
      LogError << "Invalid inputs -- Bounds: " << _splineBounds_.min
               << " to " << _splineBounds_.max
               << ", First X: " << _splineData_[0]
               << ", X spacing: " << _splineData_[1]
               << std::endl;
      for (int i=0; i<splinePointList_.size()-1; ++i) {
          double d = std::abs(splinePointList_[i].x - _splineData_[0] - i*_splineData_[1]); // offset of knot i from x0 + i*h, then as a fraction of h
          d /= _splineData_[1];
          LogError << "Invalid inputs -- point: " << i
                   << " X: " << splinePointList_[i].x
                   << " (tolerance " << d << ")"
                   << " Y: " << splinePointList_[i].y
                   << std::endl;
      }
#ifndef NDEBUG
      /// Crash if this is a release build since the inputs *should* have
      /// already been validated.  This continues during a debug build, but
      /// there will be lots of output (which I'm *sure* the user will read).
      /// @note The block is compiled only when NDEBUG is not defined: debug
      ///       builds throw here, while release builds only log the errors
      ///       and keep the dial (the opposite of the sentence above).
      LogError << "Stop execution because of invalid inputs" << std::endl;
      // Make sure that the cerr and cout output gets the message, even if the
      // log output is redirected.  This code should never execute, so it
      // might as well be loud.
      std::cout << "ERROR " << __FILE__ << "(" << __LINE__ << "): "
                << " CompactSpline: Invalid inputs" << std::endl;
      std::cerr << "ERROR " << __FILE__ << "(" << __LINE__ << "): "
                << " CompactSpline: Invalid inputs -- terminating" << std::endl;
      LogThrow("CompactSpline with invalid inputs");
#endif
  }

  for(int i=0; i<splinePointList_.size(); ++i) _splineData_[2+i] = splinePointList_[i].y;

}

/// @par Implementation
/// Clamps the input to `_splineBounds_` unless extrapolation is allowed, then
/// calls CalculateCompactSpline() with the number of knots
/// (`_splineData_.size() - 2`) as its `dim` argument. The kernel's output
/// bounds are set to -1E20 and 1E20, so it does not clamp the result.
/// Outside the knot range the kernel clamps the knot indices it reads, so
/// both slopes it uses equal the end interval's chord slope and its cubic
/// reduces to a straight line: this is the linear extrapolation.
double CompactSpline::evalResponse(const DialInputBuffer& input_) const {
  double dialInput{input_.getInputBuffer()[0]};

#ifndef NDEBUG
  LogThrowIf(not std::isfinite(dialInput), "Invalid input for CompactSpline");
#endif

  if( not _allowExtrapolation_ ){
    if     (dialInput <= _splineBounds_.min) { dialInput = _splineBounds_.min; }
    else if(dialInput >= _splineBounds_.max){ dialInput = _splineBounds_.max; }
  }

  return CalculateCompactSpline( dialInput, -1E20, 1E20, _splineData_.data(), int(_splineData_.size()-2) );
}

/// @par Implementation
/// Writes the three lines into a string stream, each prefixed with the dial
/// type name. The spline data is printed with GenericToolbox::toString().
std::string CompactSpline::getSummary() const {
  std::stringstream ss;
  ss << this->getDialTypeName() << ": spline data = " << GenericToolbox::toString(_splineData_);
  ss << std::endl << this->getDialTypeName() << ": defined bounds = { " << _splineBounds_.min << ", " << _splineBounds_.max << " }";
  ss << std::endl << this->getDialTypeName() << ": allow extrapolation ? " << _allowExtrapolation_;
  return ss.str();
};

/// @par Implementation
/// `_splineData_` holds `x0` and `h` first, then the one value per knot, so
/// the knot count is read from its size and each position is `x0 + i h`.
std::vector<double> CompactSpline::getSourceKnotPositions() const {
  if( _splineData_.empty() ){ return {}; }
  std::vector<double> positions(_splineData_.size() - 2);
  for( size_t i = 0; i < positions.size(); ++i ){
    positions[i] = _splineData_[0] + i * _splineData_[1];
  }
  return positions;
}
