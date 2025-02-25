//
// Created by Adrien Blanchet on 30/11/2022.
//

#include "RootSpline.h"


void RootSpline::buildDial(const std::vector<SplineUtils::SplinePoint>& splinePointList_){
  _spline_ = SplineUtils::buildTSpline3(splinePointList_);
}

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
