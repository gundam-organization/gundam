//
// Created by Nadrino on 23/02/2025.
//

#include "DialUtils.h"

#include <Logger.h>

#include <TDirectory.h>


namespace DialUtils{

  std::vector<DialPoint> getPointList(const TObject *src_){
    if( src_ == nullptr ) return {};

    if( src_->InheritsFrom(TGraph::Class()) ) { return getPointList((TGraph *)src_); }
    if( src_->InheritsFrom(TSpline3::Class()) ) { return getPointList((TSpline3 *)src_); }

    return {};
  }
  std::vector<DialPoint> getPointList(const TGraph *src_){
    if( src_ == nullptr ) return {};

    int nPt{src_->GetN()};

    if( nPt == 0 ){ return {};}
    if( nPt == 1 ){
      // with 1 pt don't compute SLOPES with TSpline3...
      // -> snick segfault, thank you ROOT :)
      std::vector<DialPoint> out;
      out.emplace_back();
      auto &point = out.back();
      point.x = src_->GetX()[0];
      point.y = src_->GetY()[0];
      point.slope = 0; // init
      return out;
    }

    TSpline3 spline("", src_);
    return getPointList( &spline );
  }
  std::vector<DialPoint> getPointList(const TSpline3 *src_){
    if( src_ == nullptr ) return {};
    if( src_->GetNp() == 0 ) return {};

    std::vector<DialPoint> out;
    out.reserve(src_->GetNp());

    for( int iPt = 0; iPt < src_->GetNp(); iPt++ ) {
      out.emplace_back();
      auto &point = out.back();

      // get the point coordinates
      src_->GetKnot(iPt, point.x, point.y);

      // spline is invalid if any of the points are nan
      if( not std::isfinite(point.x) or not std::isfinite(point.y) ) { return {}; }

      // make sure there's no derivative if there is only 1 point
      if( src_->GetNp() == 1 ) {
        point.slope = 0;
        continue;
      }

      // get the slope at x
      point.slope = src_->Derivative(point.x);

      // something weird happened
      if( not std::isfinite(point.slope) ) { return {}; }
    }

    return out;
  }
  std::vector<DialPoint> getPointListNoSlope(const TGraph* src_){
    if( src_ == nullptr ) { return {}; }

    int nPt{src_->GetN()};

    if( nPt == 0 ){ return {};}

    std::vector<DialPoint> out;
    out.reserve(nPt);
    for( int iPt = 0; iPt < nPt; iPt++ ){
      out.emplace_back();
      auto &point = out.back();
      point.x = src_->GetX()[iPt];
      point.y = src_->GetY()[iPt];
      point.slope = 0;
    }

    return out;
  }

  void fillCatmullRomSlopes(std::vector<DialPoint> &splinePointList_){
    // Fill the slopes with the values for a Catmull-Rom spline.
    //
    // E. Catmull and R.Rom, "A class of local interpolating splines", in
    // Barnhill, R. E.; Riesenfeld, R. F. (eds.), Computer Aided Geometric
    // Design, New York: Academic Press, pp. 317-326
    // doi:10.1016/B978-0-12-079050-0.50020-5

    if( splinePointList_.empty() ) return;

    auto nPoints{splinePointList_.size()};
    if( nPoints == 1 ) {
      splinePointList_[0].slope = 0;
      return;
    }

    // From here we have at least 2 points

    // extremity points
    size_t idx{0};
    idx = 0;
    splinePointList_[idx].slope = getSlope(splinePointList_[idx], splinePointList_[idx + 1]);
    idx = nPoints - 1;
    splinePointList_[idx].slope = getSlope(splinePointList_[idx - 1], splinePointList_[idx]);
    if( nPoints == 2 ) { return; } // stop there

    // use previous and next for the central points
    for( size_t iPt = 1; iPt < nPoints - 1; iPt++ ) {
      splinePointList_[iPt].slope = getSlope(splinePointList_[iPt - 1], splinePointList_[iPt + 1]);
    }
  }
  void fillAkimaSlopes(std::vector<DialPoint> &splinePointList_){
    // Fill the slopes with the values for an Akima spline.
    //
    // H.Akima, A New Method of Interpolation and Smooth Curve Fitting Based on
    // Local Procedures, Journal of the ACM, Volume 17, Issue 4, pp 589-602,
    // doi:10.1145/321607.321609

    if( splinePointList_.empty() ) return;

    auto nPoints(splinePointList_.size());
    if( nPoints == 1 ) {
      splinePointList_[0].slope = 0;
      return;
    }

    size_t idx{0};

    // extremity points
    idx = 0;
    splinePointList_[idx].slope = getSlope(splinePointList_[idx], splinePointList_[idx + 1]);
    idx = nPoints - 1;
    splinePointList_[idx].slope = getSlope(splinePointList_[idx - 1], splinePointList_[idx]);
    if( nPoints == 2 ) { return; } // stop there

    // Catmull-rom style
    idx = 1;
    splinePointList_[idx].slope = getSlope(splinePointList_[idx - 1], splinePointList_[idx + 1]);
    if( nPoints == 3 ) { return; } // stop there

    idx = nPoints - 2;
    splinePointList_[idx].slope = getSlope(splinePointList_[idx - 1], splinePointList_[idx + 1]);
    if( nPoints == 4 ) { return; } // stop there

    // Akima
    for( size_t iPt = 2; iPt < nPoints - 2; iPt++ ) {
      double mp1 = getSlope(splinePointList_[iPt + 1], splinePointList_[iPt + 2]);
      double m0 = getSlope(splinePointList_[iPt], splinePointList_[iPt + 1]);
      double mm1 = getSlope(splinePointList_[iPt - 1], splinePointList_[iPt]);
      double mm2 = getSlope(splinePointList_[iPt - 2], splinePointList_[iPt - 1]);
      double n1 = std::abs(mp1 - m0);
      double n0 = std::abs(mm1 - mm2);

      if( n0 > 1E-6 or n1 < 1E-6 ) { splinePointList_[iPt].slope = (n1 * mm1 + n0 * m0) / (n1 + n0); } else {
        splinePointList_[iPt].slope = 0.5 * (m0 + mm1);
      }
    }
  }
  void applyMonotonicCondition(std::vector<DialPoint> &splinePointList_){
    // Apply the Fritsh-Carlson monotonic condition to the slopes.  This
    // adjusts the slopes (when necessary), however, with Catmull-Rom the
    // modified slopes will be ignored and the monotonic criteria is
    // applied as the spline is evaluated (saves memory).
    //
    // F.N.Fritsch, and R.E.Carlson, "Monotone Piecewise Cubic
    // Interpolation" SIAM Journal on Numerical Analysis, Vol. 17, Iss. 2
    // (1980) doi:10.1137/0717021

    auto nPoints = splinePointList_.size();

    for( std::size_t i = 0; i < nPoints; i++ ) {
      double m{splinePointList_[i].slope};
      double p{splinePointList_[i].slope};

      if( i >= 1 ) { m = getSlope(splinePointList_[i - 1], splinePointList_[i]); }
      if( i < nPoints - 1 ) { p = getSlope(splinePointList_[i], splinePointList_[i + 1]); }


      double delta = std::min(std::abs(m), std::abs(p));
      // Make sure the slope at a cusp (where the average slope
      // flips sign) is zero.
      if( m * p < 0.0 ) delta = 0.0;
      // This a conservative bound on when the slope is "safe".
      // It's not the actual value where the spline becomes
      // non-monotonic.
      if( std::abs(splinePointList_[i].slope) > 3.0 * delta ) {
        splinePointList_[i].slope = 3.0 * delta;
        if( splinePointList_[i].slope < 0.0 ) { splinePointList_[i].slope *= -1; }
      }
    }
  }

  bool isFlat(const std::vector<DialPoint> &splinePointList_, double tolerance_){
    // Check if the spline is flat.  Flat functions won't be handled with
    // splines.
    if( splinePointList_.empty() or splinePointList_.size() == 1 ) return true;

    for( auto &point: splinePointList_ ) {
      // Use a tolerance based on float in case the data when through a float.
      // Keep this inside the loop so it has the right scope, and depend on the
      // compiler to do the right thing.
      if( std::abs(point.y - splinePointList_[0].y) > tolerance_ ) { return false; }
    }

    return true;
  }
  bool isUniform(const std::vector<DialPoint> &splinePointList_, double tolerance_){
    if( splinePointList_.empty() or splinePointList_.size() == 1 ) return true;

    const double avgSpace =
        (splinePointList_.back().x - splinePointList_.front().x)
        / (static_cast<double>(splinePointList_.size()) - 1.0);

    for( size_t i = 0; i < splinePointList_.size() - 1; ++i ) {
      // Find out how far the point is from the expected lattice point
      const double delta = std::abs(splinePointList_[i].x - splinePointList_[0].x - static_cast<double>(i) * avgSpace) /
                           avgSpace;
      if( delta > tolerance_ ) { return false; }
    }

    return true;
  }
  double getSlope(const DialPoint &start_, const DialPoint &end_){
    return (end_.y - start_.y) / (end_.x - start_.x);
  }

  TSpline3 buildTSpline3(const std::vector<DialPoint> &splinePointList_){
    std::vector<double> x;
    std::vector<double> y;

    auto nPoints = splinePointList_.size();
    x.reserve(nPoints);
    y.reserve(nPoints);

    for( auto splinePoint: splinePointList_ ) {
      x.emplace_back(splinePoint.x);
      y.emplace_back(splinePoint.y);
    }

    auto out = TSpline3(
    "",
    const_cast<double*>(x.data()),
    const_cast<double*>(y.data()),
    int(nPoints)
    );

    return out;
  }
}
