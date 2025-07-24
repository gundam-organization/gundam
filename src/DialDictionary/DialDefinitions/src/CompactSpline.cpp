//
// Created by Adrien Blanchet on 22/01/2023.
//

#include "CompactSpline.h"
#include "CalculateCompactSpline.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"

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
      double d = std::abs(splinePointList_[i].x - _splineData_[0] - i*_splineData_[1]);
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
          double d = std::abs(splinePointList_[i].x - _splineData_[0] - i*_splineData_[1]);
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

std::string CompactSpline::getSummary() const {
  std::stringstream ss;
  ss << this->getDialTypeName() << ": spline data = " << GenericToolbox::toString(_splineData_);
  ss << std::endl << this->getDialTypeName() << ": defined bounds = { " << _splineBounds_.min << ", " << _splineBounds_.max << " }";
  ss << std::endl << this->getDialTypeName() << ": allow extrapolation ? " << _allowExtrapolation_;
  return ss.str();
};
