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

#include <numeric>

LoggerInit([]{
  Logger::setUserHeaderStr("[SplineFactory]");
});


bool SplineDialBaseFactory::FillFromGraph(std::vector<double>& xPoint,
                                          std::vector<double>& yPoint,
                                          std::vector<double>& slope,
                                          TObject* dialInitializer,
                                          const std::string& splType) {
  if (not dialInitializer) return false;

  // Get the spline knots and slopes (starting from a graph).
  TGraph* graph = dynamic_cast<TGraph*>(dialInitializer);
  if (not graph) return false;

  // Turn the graph into a spline.
  std::string opt;
  double valBeg = 0;
  double valEnd = 0;
  if (splType == "not-a-knot") opt = "";
  else if (splType == "natural") opt = "b2,e2";
  TSpline3 spline(Form("%p", graph), graph, opt.c_str(), valBeg, valEnd);

  xPoint.reserve(spline.GetNp());
  yPoint.reserve(spline.GetNp());
  slope.reserve(spline.GetNp());
  xPoint.clear();
  yPoint.clear();
  slope.clear();
  for (int i = 0; i<graph->GetN(); ++i) {
    double x; double y;
    spline.GetKnot(i,x,y);
    if (!std::isfinite(x)) return false;
    if (!std::isfinite(y)) return false;
    double d = spline.Derivative(x);
    if (!std::isfinite(d)) return false;
    xPoint.push_back(x);
    yPoint.push_back(y);
    slope.push_back(d);
  }

  return true;
}

bool SplineDialBaseFactory::FillFromSpline(std::vector<double>& xPoint,
                                           std::vector<double>& yPoint,
                                           std::vector<double>& slope,
                                           TObject* dialInitializer,
                                           const std::string& splType) {
  if (not dialInitializer) return false;

  // Get the spline knots and slopes (starting from a spline).
  TSpline3* spline = dynamic_cast<TSpline3*>(dialInitializer);
  if (not spline) return false;

  xPoint.reserve(spline->GetNp());
  yPoint.reserve(spline->GetNp());
  slope.reserve(spline->GetNp());
  xPoint.clear();
  yPoint.clear();
  slope.clear();
  for (int i = 0; i<spline->GetNp(); ++i) {
    double x; double y;
    spline->GetKnot(i,x,y);
    if (!std::isfinite(x)) return false;
    if (!std::isfinite(y)) return false;
    double d = spline->Derivative(x);
    if (!std::isfinite(d)) return false;
    xPoint.push_back(x);
    yPoint.push_back(y);
    slope.push_back(d);
  }

  return true;
}

void SplineDialBaseFactory::MakeMonotonic(const std::vector<double>& xPoint,
                                          const std::vector<double>& yPoint,
                                          std::vector<double>& slope) {
  // Apply the monotonic condition to the slopes.  This always adjusts the
  // slopes, however, with Catmull-Rom the modified slopes will be ignored
  // and the monotonic criteria is applied as the spline is evaluated.
  for (int i = 0; i<xPoint.size(); ++i) {
    double m{std::numeric_limits<double>::infinity()};
    if (i>0) m = (yPoint[i] - yPoint[i-1])/(xPoint[i] - xPoint[i-1]);
    double p{std::numeric_limits<double>::infinity()};
    if (i<xPoint.size()-1) p =(yPoint[i+1]-yPoint[i])/(xPoint[i+1]-xPoint[i]);
    double delta = std::min(std::abs(m),std::abs(p));
    // This applies an upper bound on when the slope is "safe".  It's not
    // the actual bound for when the actual bound on when the slope makes
    // the spline non-monotonic.
    if (std::abs(slope[i]) > 3.0*delta) {
      if (slope[i] < 0.0) slope[i] = -3.0*delta;
      slope[i] = 3.0*delta;
    }
  }
}

DialBase* SplineDialBaseFactory::operator () (const std::string& dialType_,
                                              const std::string& dialSubType_,
                                              TObject* dialInitializer_,
                                              bool useCachedDial_) {

  if (dialInitializer_ == nullptr) return nullptr;

  // The types of splines are "not-a-knot", "natural", "catmull-rom", and
  // "ROOT".  The "not-a-knot" spline will give the same curve as ROOT (and
  // might be implemented with at TSpline3).  The "ROOT" spline will use an
  // actual TSpline3 (and is slow).  The natural and catmull-rom splines are
  // just as expected.
  std::string splType = "not-a-knot";  // The default.
  if (dialSubType_.find("not-a-knot") != std::string::npos) splType = "not-a-knot";
  if (dialSubType_.find("catmull") != std::string::npos) splType = "catmull-rom";
  if (dialSubType_.find("natural") != std::string:: npos) splType = "natural";
  if (dialSubType_.find("ROOT") != std::string:: npos) splType = "ROOT";

  std::vector<double> xPoints;
  std::vector<double> yPoints;
  std::vector<double> slopePoints;

  if      ( FillFromGraph(xPoints, yPoints, slopePoints, dialInitializer_, splType) ) { /* filled from TGraph successful */ }
  else if ( FillFromSpline(xPoints, yPoints, slopePoints, dialInitializer_, splType) ) { /* filled from TSpline3 successful */ }
  else{ return nullptr; }

  // Check that there are enough points in the spline.
  if (xPoints.size() < 2) {
    LogWarning << "Splines must have at least two points." << std::endl;
    return nullptr;
  }

  // If there are only two points, then force catmull-rom
  if (xPoints.size() < 3) { splType = "catmull-rom"; }

  // Check that there are equal numbers of X and Y
  LogThrowIf(xPoints.size() != yPoints.size(), "INVALID Spline: must have the same number of X and Y points");

  // Check that the X points are in increasing order.
  double lastX{std::nan("")};
  for (double xPoint : xPoints) {
    LogThrowIf(xPoint <= lastX, "INVALID Spline: points are not in increasing order.");
    lastX = xPoint;
  }

  // Check that the spline isn't flat and 1.0
  bool isFlat{true};
  double lastY{std::nan("")};
  for( double yPoint : yPoints ){
    if( yPoint == lastY ){ isFlat = false; break; }
    lastY = yPoint;
  }
  if ( isFlat ){
    // get rid of the spline if the flat response is one
    if( lastY == 1. ){
      return nullptr;
    }
  }

  ////////////////////////////////////////////////////////////////
  // Condition the slopes as necessary (only matters if the dial sub-type
  // includes "monotonic"
  ////////////////////////////////////////////////////////////////
  bool isMonotonic = ( dialSubType_.find("monotonic") != std::string::npos );  // So the right catmull-rom class can be chosen.
  if ( isMonotonic ) { MakeMonotonic(xPoints, yPoints, slopePoints); }

  ////////////////////////////////////////////////////////////////
  // Check if the spline can be treated as having uniformly spaced knots.
  ////////////////////////////////////////////////////////////////
  double s = (xPoints.back() - xPoints.front()) / (double(xPoints.size()) - 1.0);
  bool hasUniformlySpacedKnots = true;
  for (size_t iPoint=1; iPoint < xPoints.size(); ++iPoint) {
    if (std::abs(((xPoints[iPoint] - xPoints[iPoint - 1]) - s) / s) > 5E-6) {
      hasUniformlySpacedKnots = false; break;
    }
  }

  // Stuff the created dial into a unique_ptr, so it will be properly deleted
  // in the event of an exception.
  std::unique_ptr<DialBase> dialBase;

  // Create the right low level spline class.
  if (splType == "catmull-rom") {
    LogThrowIf(not hasUniformlySpacedKnots, "Catmull-rom splines need a uniformly spaced points");
    if (not isMonotonic) {
      (useCachedDial_ ? ( dialBase = std::make_unique<CompactSplineCache>() ) : dialBase = std::make_unique<SplineCache>() );
    }
    else {
      (useCachedDial_ ? ( dialBase = std::make_unique<MonotonicSplineCache>() ) : dialBase = std::make_unique<MonotonicSpline>() );
    }
  }
  else if (splType == "ROOT") {
    dialBase = std::make_unique<Spline>();
  }
  else {
    if (not hasUniformlySpacedKnots) {
      (useCachedDial_ ? ( dialBase = std::make_unique<GeneralSplineCache>() ) : dialBase = std::make_unique<GeneralSpline>() );
    }
    else {
      (useCachedDial_ ? ( dialBase = std::make_unique<UniformSplineCache>() ) : dialBase = std::make_unique<UniformSpline>() );
    }
  }

  // Initialize the spline
  dialBase->buildDial(xPoints, yPoints, slopePoints);

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