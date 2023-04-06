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

#include "Logger.h"

#include "TGraph.h"
#include "TSpline.h"

#include <memory>
#include <numeric>

LoggerInit([]{
  Logger::setUserHeaderStr("[SplineDialBaseFactory]");
});

SplineDialBaseFactory::SplineDialBaseFactory() = default;
SplineDialBaseFactory::~SplineDialBaseFactory() = default;

bool SplineDialBaseFactory::FillFromGraph(std::vector<double>& xPoint,
                                          std::vector<double>& yPoint,
                                          std::vector<double>& slope,
                                          TObject* dialInitializer,
                                          const std::string& splType) {
  if (not dialInitializer) return false;

  // Get the spline knots and slopes (starting from a graph).
  auto* graph = dynamic_cast<TGraph*>(dialInitializer);
  if (graph == nullptr) return false;

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
  auto* spline = dynamic_cast<TSpline3*>(dialInitializer);
  if (spline == nullptr) return false;

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

DialBase* SplineDialBaseFactory::operator () (const std::string& dialType,
                                              const std::string& dialSubType,
                                              TObject* dialInitializer,
                                              bool cached) {

  if (not dialInitializer) return nullptr;

  // The types of splines are "not-a-knot", "natural", "catmull-rom", and
  // "ROOT".  The "not-a-knot" spline will give the same curve as ROOT (and
  // might be implemented with at TSpline3).  The "ROOT" spline will use an
  // actual TSpline3 (and is slow).  The natural and catmull-rom splines are
  // just as expected.
  std::string splType = "not-a-knot";  // The default.

  if ( dialSubType.find("not-a-knot")!=std::string::npos ) splType = "not-a-knot";
  if ( dialSubType.find("catmull")!=std::string::npos ) splType = "catmull-rom";
  if ( dialSubType.find("natural") != std::string:: npos ) splType = "natural";
  if ( dialSubType.find("ROOT") != std::string:: npos ) splType = "ROOT";

  LogDebug << GET_VAR_NAME_VALUE(splType) << std::endl;

  std::vector<double> xPoint;
  std::vector<double> yPoint;
  std::vector<double> slope;

  do {
    if (FillFromGraph(xPoint,yPoint,slope,dialInitializer,splType)) break;
    if (FillFromSpline(xPoint,yPoint,slope,dialInitializer,splType)) break;
    return nullptr;
  } while (false);

  // Check that there are enough points in the spline.
  if (xPoint.size() < 2) {
    LogWarning << "Splines must have at least two points." << std::endl;
    return nullptr;
  }

  // If there are only two points, then force catmull-rom
  if (xPoint.size()<3) {
    splType = "catmull-rom";
  }

  LogDebug << "CHANGED: " << GET_VAR_NAME_VALUE(splType) << std::endl;

  // Check that there are equal numbers of X and Y
  if (xPoint.size() != yPoint.size()) {
    LogWarning << "Splines must have the same number of X and Y points"
               << std::endl;
    return nullptr;
  }

  // Check that the X points are in increasing order.
  double lastX{std::nan("")};
  for (int i = 0; i<xPoint.size(); ++i) {
    if (xPoint[i] <= lastX) return nullptr;
  }

  // Check that the spline isn't flat and 1.0
  bool flat{true};
  double lastY{std::nan("")};
  for (int i = 0; i<xPoint.size(); ++i) {
    if (std::abs(yPoint[i]-lastY) > 1E-6) flat = false;
    lastY = yPoint[i];
  }
  if (flat && std::abs(lastY-1.0)) return nullptr;

  ////////////////////////////////////////////////////////////////
  // Condition the slopes as necessary (only matters if the dial sub-type
  // includes "monotonic"
  ////////////////////////////////////////////////////////////////
  bool monotonic = false;  // So the right catmull-rom class can be chosen.
  if (dialSubType.find("monotonic") != std::string::npos) {
    MakeMonotonic(xPoint,yPoint,slope);
    monotonic = true;
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

  // Create the right low level spline class.
  if (splType == "catmull-rom") {
    LogThrowIf(not uniform,
               "Catmull-rom splines need a uniformly spaced points");
    if (not monotonic) {
      dialBase.reset(
        (not cached) ? new CompactSpline: new CompactSplineCache);
    }
    else {
      dialBase.reset(
        (not cached) ? new MonotonicSpline: new MonotonicSplineCache);
    }
  }
  else if (splType == "ROOT") {
    dialBase = std::make_unique<Spline>();
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

  // Initialize the spline
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
