#include "SplineDialBaseFactory.h"

#include "Spline.h"
#include "GeneralSpline.h"
#include "UniformSpline.h"
#include "CompactSpline.h"
#include "MonotonicSpline.h"

#include "SplineCache.h"
#include "GeneralSplineCache.h"
#include "UniformSplineCache.h"
#include "CompactSplineCache.h"
#include "MonotonicSplineCache.h"

#include "TGraph.h"
#include "TSpline.h"

SplineDialBaseFactory::SplineDialBaseFactory() {}
SplineDialBaseFactory::~SplineDialBaseFactory() {}

DialBase* SplineDialBaseFactory::operator () (std::string dialType,
                                              std::string dialSubType,
                                              TObject* dialInitializer,
                                              bool cached) {

  // The types of splines are "not-a-knot", "natural", "catmull-rom", and
  // "ROOT".  The "not-a-knot" spline will give the same curve as ROOT (and
  // might be implemented with at TSpline3).  The "ROOT" spline will use an
  // actual TSpline3 (and is slow).  The natural and catmull-rom splines are
  // just as expected.
  std::string splType = "not-a-knot";
  if (dialSubType.find("catmull")!=std::string::npos) splType = "catmull-rom";
  if (dialSubType.find("natural") != std::string:: npos) splType = "natural";
  if (dialSubType.find("ROOT") != std::string:: npos) splType = "ROOT";

  std::vector<double> xPoint;
  std::vector<double> yPoint;
  std::vector<double> slope;

  do {
    // Get the spline knots and slopes (starting from a graph).
    TGraph* graph = dynamic_cast<TGraph*>(dialInitializer);
    if (graph) {
      std::string opt;
      double valBeg = 0;
      double valEnd = 0;
      if (splType == "natural") opt = "b2,e2";
      TSpline3 spline(Form("%p", graph), graph, opt.c_str(), valBeg, valEnd);
      xPoint.reserve(spline.GetNp());
      yPoint.reserve(spline.GetNp());
      slope.reserve(spline.GetNp());
      for (int i = 0; i<graph->GetN(); ++i) {
        double x; double y;
        spline.GetKnot(i,x,y);
        double d = spline.Derivative(x);
        xPoint.push_back(x);
        yPoint.push_back(y);
        slope.push_back(d);
      }
      break;
    }

    // Get the spline knots and slopes (starting from a spline).
    TSpline3* spline = dynamic_cast<TSpline3*>(dialInitializer);
    if (spline) {
      xPoint.reserve(spline->GetNp());
      yPoint.reserve(spline->GetNp());
      slope.reserve(spline->GetNp());
      for (int i = 0; i<spline->GetNp(); ++i) {
        double x; double y;
        spline->GetKnot(i,x,y);
        double d = spline->Derivative(x);
        xPoint.push_back(x);
        yPoint.push_back(y);
        slope.push_back(d);
      }
      break;
    }

    LogThrow("dialInitialize must be a TGraph or a TSpline3");
  } while (false);

  LogThrowIf(xPoint.size() < 2,
             "Splines must have at least two points.");
  LogThrowIf(xPoint.size() != yPoint.size(),
             "Splines must have the same number of X and Y points");

  ////////////////////////////////////////////////////////////////
  // Condition the slopes as necessary (only matters if the dial sub-type
  // includes "monotonic"
  ////////////////////////////////////////////////////////////////
  bool monotonic = false;
  if (dialSubType.find("monotonic") != std::string::npos) {
    monotonic = true;
    // Apply the monotonic condition to the slopes.  This always adjusts the
    // slopes, however, with Catmull-Rom the modified slopes will be ignored
    // and the monotonic criteria is applied as the spline is evaluated.

    /// DUMMY FOR NOW!!!!!
  }

  ////////////////////////////////////////////////////////////////
  // Check if the spline can be treated as having uniformly spaced knots.
  ////////////////////////////////////////////////////////////////
  double s = (xPoint.back() - xPoint.front())/(xPoint.size()-1);
  bool uniform = true;
  for (int i=1; i<xPoint.size(); ++i) {
    if (std::abs(xPoint[i]-xPoint[i-1]-s) > 1E-6) {
      uniform = false;
      break;
    }
  }

  // Stuff the created dial into a unique_ptr, so it will be properly deleted
  // in the event of an exception.
  std::unique_ptr<DialBase> dialBase;

  // Create the right low level spline.
  if (splType == "catmull-rom") {
    LogThrowIf(not uniform,
               "Catmull-rom splines need a uniformly spaced points");
    if (monotonic) {
      dialBase.reset(
        (not cached) ? new MonotonicSpline: new MonotonicSplineCache);
    }
    else {
      dialBase.reset(
        (not cached) ? new CompactSpline: new CompactSplineCache);
    }
  }
  else if (splType == "ROOT") {
    dialBase.reset(new Spline);
  }
  else {
    if (not uniform) {
      dialBase.reset(
        (not cached) ? new GeneralSpline: new GeneralSplineCache);
    }
    else {
      dialBase.reset(
        (not cached) ? new UniformSpline: new UniformSplineCache);
    }
  }

  dialBase->buildDial(xPoint,yPoint,slope);

  // Pass the ownership without any constraints!
  return dialBase.release();
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
