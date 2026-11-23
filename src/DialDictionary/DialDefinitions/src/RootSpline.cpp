//
// Created by Adrien Blanchet on 30/11/2022.
//

/// @file RootSpline.cpp
/// @brief Implements RootSpline: builds and evaluates a ROOT TSpline3.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in RootSpline.h. The
/// comments here describe how each method does its job.

#include "RootSpline.h"

/// @par Implementation
/// Delegates to DialUtils::buildTSpline3(), which copies the `x` and `y` of
/// each point into arrays and calls the TSpline3 constructor without
/// end-condition options. The returned spline is copied into `_spline_`.

void RootSpline::buildDial(const std::vector<DialUtils::DialPoint>& splinePointList_){
  _spline_ = DialUtils::buildTSpline3(splinePointList_);
}

/// @par Implementation
/// When extrapolation is not allowed and the input is at or beyond an end
/// knot, returns TSpline3::Eval() at that end knot (TSpline3::GetXmin() or
/// TSpline3::GetXmax()) instead of clamping the input variable. Otherwise
/// evaluates the spline at the input.
double RootSpline::evalResponse(const DialInputBuffer& input_) const {
  const double dialInput{input_.getInputBuffer()[0]};

#ifndef NDEBUG
  LogThrowIf(not std::isfinite(dialInput), "Invalid input for Spline");
#endif

  if( not _allowExtrapolation_ ){
    if (dialInput <= _spline_.GetXmin()) {
      return _spline_.Eval( _spline_.GetXmin() );
    }
    else if (dialInput >= _spline_.GetXmax()) {
      return _spline_.Eval( _spline_.GetXmax() );
    }
  }
  return _spline_.Eval( dialInput );
}

//  A Lesser GNU Public License

//  Copyright (C) 2023 GUNDAM DEVELOPERS

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA

// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:

/// @par Implementation
/// Reads each knot with TSpline3::GetKnot() and keeps its `x`.
std::vector<double> RootSpline::getSourceKnotPositions() const {
  std::vector<double> positions(_spline_.GetNp());
  for( size_t i = 0; i < positions.size(); ++i ){
    double y;
    _spline_.GetKnot(int(i), positions[i], y);
  }
  return positions;
}
