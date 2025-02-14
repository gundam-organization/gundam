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

#include "GraphDialBaseFactory.h"

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

DialBase* SplineDialBaseFactory::makeDial(const std::string& dialTitle_,
                                          const std::string& dialType_,
                                          const std::string& dialSubType_,
                                          TObject* dialInitializer_,
                                          bool useCachedDial_) {

  if (dialInitializer_ == nullptr) return nullptr;

  // The types of cubic splines are "not-a-knot", "natural", "catmull-rom",
  // and "ROOT".  The "not-a-knot" spline will give the same curve as ROOT
  // (and might be implemented with at TSpline3).  The "ROOT" spline will use
  // an actual TSpline3 (and is slow).  The "natural" and "catmull-rom"
  // splines are just as expected (you can see the underlying math on
  // Wikipedia or another source).  Be careful about the order since later
  // conditionals can override earlier ones.
  std::string splType = "not-a-knot";  // The default.
  if (dialSubType_.find("akima") != std::string::npos) splType = "akima";
  if (dialSubType_.find("catmull") != std::string::npos) splType = "catmull-rom";
  if (dialSubType_.find("natural") != std::string::npos) splType = "natural";
  if (dialSubType_.find("not-a-knot") != std::string::npos) splType = "not-a-knot";
  if (dialSubType_.find("pixar") != std::string::npos) {
    splType = "catmull-rom";
    // sneaky output... logger would tattle on me.
    static bool woody=true;
    if (woody) std::cout << std::endl << std::endl << "You got a friend in me!" << std::endl;
    woody=false;
  }
  if (dialSubType_.find("ROOT") != std::string::npos) splType = "ROOT";

  // Get the numeric tolerance for when a uniform spline can be used.  We
  // should be able to set this in the DialSubType.
  const double defUniformityTolerance{16*std::numeric_limits<float>::epsilon()};
  double uniformityTolerance{defUniformityTolerance};
  if (dialSubType_.find("uniformity(") != std::string::npos) {
    std::size_t bg = dialSubType_.find("uniformity(");
    bg = dialSubType_.find("(",bg);
    std::size_t en = dialSubType_.find(")",bg);
    LogThrowIf(en == std::string::npos,
               "Invalid spline uniformity with dialSubType: " << dialSubType_
               << " dial: " << dialTitle_);
    en = en - bg;
    std::string uniformityString = dialSubType_.substr(bg+1,en-1);
    std::istringstream unif(uniformityString);
    unif >> uniformityTolerance;
  }

  _xPointListBuffer_.clear();
  _yPointListBuffer_.clear();
  _slopeListBuffer_.clear();

  ///////////////////////////////////////////////////////////////////////
  // Side-effect programming alert.  The conditionals are doing the actual
  // work and setting xPoint, yPoint and slopePoint.  We only have a couple
  // of types of input for now, but watch out for the evil
  // if-then-elseif-elseif-elseif-elseif idiom.  Change this to do-while-false
  // if the number of kinds of initializers is more than a few.
  ///////////////////////////////////////////////////////////////////////
  if (FillFromGraph(_xPointListBuffer_, _yPointListBuffer_, _slopeListBuffer_,
                     dialInitializer_, splType)) {
    // The points were from a ROOT graph like object and the points were
    // filled, don't check any further.
  }
  else if (FillFromSpline(_xPointListBuffer_, _yPointListBuffer_, _slopeListBuffer_,
                          dialInitializer_, splType) ) {
    // The points were from a ROOT TSpline3 like object and the points were
    // filled, don't check any further.
  }
  else {
    // If we get to this point, we don't know how to grab the spline points
    // out of the initializer, so flag this as an invalid dial.
    return nullptr;
  }

  // Check that we got at least some points!  A single point will be treated
  // as a constant value, but it's not an error.
  if (_xPointListBuffer_.empty()) {
    LogAlertOnce << "Splines must have at least one point." << std::endl;
    return nullptr;
  }

  // Check that there are equal numbers of X and Y.  There have to be an equal
  // number of points or something is very wrong.
  LogThrowIf( _xPointListBuffer_.size() != _yPointListBuffer_.size(),
              "INVALID Spline Input: "
              << "must have the same number of X and Y points "
              << "for dial " << dialTitle_ );

  // Check that the X points are in strictly increasing order (sorted order is
  // not sufficient) with explicit comparisons in case we want to add more
  // detailed logic.  This shouldn't happen, and indicates there is a problem
  // with the inputs.  Don't try to continue!
  for (int i = 1; i<_xPointListBuffer_.size(); ++i) {
    LogThrowIf(_xPointListBuffer_[i] <= _xPointListBuffer_[i-1],
               "INVALID Spline Input: points are not in increasing order "
               << " for dial " << dialTitle_);
  }

  // Check if the spline is flat.  Flat functions won't be handled with
  // splines.
  bool isFlat{true};
  for (double y : _yPointListBuffer_) {
    // Use a tolerance based on float in case the data when through a float.
    // Keep this inside the loop so it has the right scope, and depend on the
    // compiler to do the right thing.
    const double toler{2*std::numeric_limits<float>::epsilon()};
    const double delta{std::abs(y-_yPointListBuffer_[0])};
    if (delta > toler) isFlat = false;
  }

  // If the function is flat AND equal to one, the drop it.  Compare against
  // float accuracy in case the value was actually calculated against with a
  // float.
  if (std::abs(_yPointListBuffer_[0]-1.0)
      < 2*std::numeric_limits<float>::epsilon()
      and isFlat) {
    return nullptr;
  }

  // If the function is equal to a constant (but not to one) then we don't
  // need a Spline so use the faster "Shift" dial (which applies a "scale"
  // factor, and not an additive shift).
  if (isFlat) {
    // Do the unique_ptr dance in case there are exceptions.
    std::unique_ptr<DialBase> dialBase = std::make_unique<Shift>();
    dialBase->buildDial(_yPointListBuffer_[0]);
    return dialBase.release();
  }

#define SHORT_CIRCUIT_SMALL_SPLINES
#ifdef  SHORT_CIRCUIT_SMALL_SPLINES
  if (_xPointListBuffer_.size() < 3) {
    GraphDialBaseFactory grapher;
    return grapher.makeDial(dialTitle_,
                            "Graph","",
                            dialInitializer_,
                            useCachedDial_);
  }
#endif

  // Sanity check.  By the time we get here, there can't be fewer than two
  // points, and it should have been trapped above by other conditionals
  // (e.g. a single point "spline" should have been flagged as flat).
  LogThrowIf((_xPointListBuffer_.size() < 2),
             "Input data logic error: two few points "
             << "for dial " << dialTitle_ );

  ////////////////////////////////////////////////////////////////
  // Check if the spline slope calculation should be updated.  The slopes for
  // not-a-knot and natural splines are calculated by FillFromGraph and
  // FillFromSpoline using ROOT code.  That means we need to fill in the
  // slopes for the other types ("catmull-rom", "akima")
  if (splType == "catmull-rom") {
    // Fill the slopes according to the Catmull-Rom prescription.
    FillCatmullRomSlopes(_xPointListBuffer_,
                         _yPointListBuffer_,
                         _slopeListBuffer_);
  }
  else if (splType == "akima") {
    // Fill the slopes according to the Akima prescription.
    FillAkimaSlopes(_xPointListBuffer_, _yPointListBuffer_, _slopeListBuffer_);
  }

  ////////////////////////////////////////////////////////////////
  // Check if the spline can be treated as having uniformly spaced knots.
  ////////////////////////////////////////////////////////////////
  bool isUniform = true;
  for (int i=0; i<_xPointListBuffer_.size()-1; ++i) {
    // Could be precalculated, but this only gets run once per dial so go for
    // clarity instead.  The compiler probably optimizes it out of the loop.
    const double avgSpace = (_xPointListBuffer_.back()-_xPointListBuffer_.front())/(_xPointListBuffer_.size()-1.0);
    // Find out how far the point is from the expected lattice point
    const double delta = std::abs(_xPointListBuffer_[i] - _xPointListBuffer_[0] - i*avgSpace)/avgSpace;
    if (delta < uniformityTolerance) continue;
    // Point isn't in the right place so this is not uniform and break out of
    // the loop.
    isUniform = false;
    break;
  }

  ////////////////////////////////////////////////////////////////
  // Check if the spline is suppose to be monotonic and condition the slopes
  // if necessary.  This is ignored by "ROOT" splines.  The Catmull-Rom
  // splines have a special implementation for monotonic splines, so save a
  // flag that can be checked later.
  ////////////////////////////////////////////////////////////////
  bool isMonotonic = ( dialSubType_.find("monotonic") != std::string::npos );
  if ( isMonotonic ) { ::util::MakeMonotonicSpline(_xPointListBuffer_, _yPointListBuffer_, _slopeListBuffer_); }

  // If there are only two points, then force a catmull-rom.  This could be
  // handled using a graph, but Catmull-Rom is fast, and works better with the
  // GPU.  The isMonotonic is forced to false so that this uses CompactSpline
  // instead of MonotonicSpline.
  if (_xPointListBuffer_.size() < 3) {
    splType = "catmull-rom";
    isMonotonic = false;
  }

  ///////////////////////////////////////////////////////////
  // Create the right kind low level spline class base on all of the previous
  // queries.  This is pushing the if-elseif-elseif limit so watch for when it
  // should change to the do-while-false idiom.  Make sure the individual
  // conditionals are less than 10 lines.
  ///////////////////////////////////////////////////////////
  std::unique_ptr<DialBase> dialBase;
  if (splType == "ROOT") {
    // The ROOT implementation of the spline has been explicitly requested, so
    // use it.
    dialBase = std::make_unique<Spline>();
  }
  else if (splType == "catmull-rom" and isMonotonic) {
    // Catmull-Rom is handled as a special case because it ignores the slopes,
    // and has an explicit monotonic implementatino.  It also must have
    // uniformly spaced knots.
    if (not isUniform) {
      LogError << "Monotonic Catmull-rom splines need a uniformly spaced points"
               << " Dial: " << dialTitle_
               << std::endl;
      double step = (_xPointListBuffer_.back()-_xPointListBuffer_.front())/(_xPointListBuffer_.size()-1);
      for (int i = 0; i<_xPointListBuffer_.size()-1; ++i) {
        LogError << i << " --  X: " << _xPointListBuffer_[i]
                 << " X+1: " << _xPointListBuffer_[i+1]
                 << " step: " << step
                 << " error: " << _xPointListBuffer_[i+1] - _xPointListBuffer_[i] - step
                 << std::endl;
      }
      // If the user specified a tolerance then crash, otherwise trust the
      // user knows that it's not uniform and continue.
      LogThrowIf(uniformityTolerance != defUniformityTolerance,
                 "Invalid catmull-rom inputs -- Nonuniform spacing");
    }
    dialBase = (not useCachedDial_) ?
      std::make_unique<MonotonicSpline>():
      std::make_unique<MonotonicSplineCache>();
  }
  else if (splType == "catmull-rom") {
    // Catmull-Rom is handled as a special case because it ignores the slopes.
    // This is the version when the spline doesn't need to be monotonic.
    if (not isUniform) {
      LogError << "Catmull-rom splines need a uniformly spaced points"
               << " Dial: " << dialTitle_
               << std::endl;
      double step = (_xPointListBuffer_.back()-_xPointListBuffer_.front())/(_xPointListBuffer_.size()-1);
      for (int i = 0; i<_xPointListBuffer_.size()-1; ++i) {
        LogError << i << " --  X: " << _xPointListBuffer_[i]
                 << " X+1: " << _xPointListBuffer_[i+1]
                 << " step: " << step
                 << " error: " << _xPointListBuffer_[i+1] - _xPointListBuffer_[i] - step
                 << std::endl;
      }
      // If the user specified a tolerance then crash, otherwise trust the
      // user knows that it's not uniform and continue.
      LogThrowIf(uniformityTolerance != defUniformityTolerance,
                 "Invalid catmull-rom inputs -- Nonuniform spacing");
    }
    dialBase = (not useCachedDial_) ?
      std::make_unique<CompactSpline>():
      std::make_unique<CompactSplineCache>();
  }
  else if (isUniform) {
    // Haven't matched a specific special case, but we have uniformly spaced
    // knots so we can use the faster UniformSpline implementation.
    dialBase = (not useCachedDial_) ?
      std::make_unique<UniformSpline>():
      std::make_unique<UniformSplineCache>();
  }
  else {
    // Haven't matched a specific special case, and the knots are not
    // uniformly spaced, so we have to use the GeneralSpline implemenatation
    // which can handle any kind of cubic spline.
    dialBase = (not useCachedDial_) ?
      std::make_unique<GeneralSpline>():
      std::make_unique<GeneralSplineCache>();
  }

  // Initialize the spline from the slopes
  dialBase->buildDial(_xPointListBuffer_, _yPointListBuffer_, _slopeListBuffer_);

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
