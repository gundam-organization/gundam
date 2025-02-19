#include "SplineDialBaseFactory.h"

// Explicitly list the headers that are actually needed.  Do not include
// others.
#include "Spline.h"
#include "SimpleSpline.h"
#include "CompactSpline.h"
#include "UniformSpline.h"
#include "GeneralSpline.h"
#include "MonotonicSpline.h"
#include "MakeMonotonicSpline.h"
#include "Shift.h"

#include "TGraph.h"
#include "TSpline.h"

#include <limits>


bool SplineDialBaseFactory::FillFromGraph(std::vector<double>& xPoint,
                                          std::vector<double>& yPoint,
                                          std::vector<double>& slope,
                                          TObject* dialInitializer,
                                          const std::string& splType) {
  if (dialInitializer == nullptr) return false;

  // Get the spline knots and slopes (starting from a graph).
  TGraph* graph = dynamic_cast<TGraph*>(dialInitializer);
  if ( graph == nullptr ) return false;

  if ( graph->GetN() == 1 ){
    xPoint.reserve(1); xPoint.clear(); xPoint.emplace_back( graph->GetX()[0] );
    yPoint.reserve(1); yPoint.clear(); yPoint.emplace_back( graph->GetY()[0] );
    slope.reserve(1); slope.clear(); slope.emplace_back( 0 );
    return true;
  }

  // Turn the graph into a spline.  This is where we can handle getting the
  // slopes for not-a-knot and natural splines (by using ROOT TSpline3
  // boundary conditinos).
  std::string opt;
  double valBeg = 0;
  double valEnd = 0;
  if      ( splType == "not-a-knot" ) opt = ""; // be explicit!
  else if ( splType == "natural" ) opt = "b2,e2"; // fix second derivative.
  TSpline3 spline(Form("%p", graph), graph, opt.c_str(), valBeg, valEnd);

  xPoint.reserve(spline.GetNp());
  yPoint.reserve(spline.GetNp());
  slope.reserve(spline.GetNp());
  xPoint.clear();
  yPoint.clear();
  slope.clear();

  // Copy the points, values, and slopes to the output, but also check that
  // we're getting valid numeric values.
  for (int i = 0; i<graph->GetN(); ++i) {
    double x; double y;
    spline.GetKnot(i,x,y);
    if (!std::isfinite(x)) return false;
    if (!std::isfinite(y)) return false;
    double d = spline.Derivative(x);
    if (!std::isfinite(d)) return false;
    xPoint.emplace_back(x);
    yPoint.emplace_back(y);
    slope.emplace_back(d);
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
  if (spline == nullptr) return false;

  // We could apply logic similar to FillFromGraph to implement not-a-knot and
  // natural splines, but if the analysis is actually saving splines in the
  // input files, then assume it knows what it is doing.  Directly use the
  // saved spline.
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
    xPoint.emplace_back(x);
    yPoint.emplace_back(y);
    slope.emplace_back(d);
  }

  return true;
}

void SplineDialBaseFactory::FillCatmullRomSlopes(
  const std::vector<double>& xPoint,
  const std::vector<double>& yPoint,
  std::vector<double>& slope) {
  // Fill the slopes with the values for a Catmull-Rom spline.
  //
  // E. Catmull and R.Rom, "A class of local interpolating splines", in
  // Barnhill, R. E.; Riesenfeld, R. F. (eds.), Computer Aided Geometric
  // Design, New York: Academic Press, pp. 317-326
  // doi:10.1016/B978-0-12-079050-0.50020-5

  if (xPoint.size() < 1) return;
  if (xPoint.size() < 2) slope.front() = 0.0;

  int k = 0;
  slope[k] = (yPoint[k+1]-yPoint[k])/(xPoint[k+1]-xPoint[k]);
  for (int i = 1; i<xPoint.size()-1; ++i) {
    slope[i] = (yPoint[i+1] - yPoint[i-1])/(xPoint[i+1]-xPoint[i-1]);
  }
  k = yPoint.size()-1;
  slope[k] = (yPoint[k]-yPoint[k-1])/(xPoint[k]-xPoint[k-1]);
}

void SplineDialBaseFactory::FillAkimaSlopes(
  const std::vector<double>& xPoint,
  const std::vector<double>& yPoint,
  std::vector<double>& slope) {
  // Fill the slopes with the values for an Akima spline.
  //
  // H.Akima, A New Method of Interpolation and Smooth Curve Fitting Based on
  // Local Procedures, Journal of the ACM, Volume 17, Issue 4, pp 589-602,
  // doi:10.1145/321607.321609

  if (xPoint.size() < 1) return;
  if (xPoint.size() < 2) slope.front() = 0.0;

  int k = 0;
  slope[k] = (yPoint[k+1]-yPoint[k])/(xPoint[k+1]-xPoint[k]);
  if (xPoint.size() > 2) {
    k = 1;
    slope[k] = (yPoint[k+1]-yPoint[k-1])/(xPoint[k+1]-xPoint[k-1]);
  }

  for (int i = 2; i<xPoint.size()-2; ++i) {
    double mp1 = (yPoint[i+2] - yPoint[i+1])/(xPoint[i+2]-xPoint[i+1]);
    double m0 = (yPoint[i+1] - yPoint[i])/(xPoint[i+1]-xPoint[i]);
    double mm1 = (yPoint[i] - yPoint[i-1])/(xPoint[i]-xPoint[i-1]);
    double mm2 = (yPoint[i-1] - yPoint[i-2])/(xPoint[i-1]-xPoint[i-2]);
    double n1 = std::abs(mp1-m0);
    double n0 = std::abs(mm1-mm2);

    if (n0 > 1E-6 or n1 < 1E-6) {
      slope[i] = (n1*mm1 + n0*m0)/(n1+n0);
    }
    else {
      slope[i] = 0.5*(m0+mm1);
    }
  }

  if (xPoint.size() > 2) {
    k = yPoint.size()-2;
    slope[k] = (yPoint[k+1]-yPoint[k-1])/(xPoint[k+1]-xPoint[k-1]);
  }
  k = yPoint.size()-1;
  slope[k] = (yPoint[k]-yPoint[k-1])/(xPoint[k]-xPoint[k-1]);
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
