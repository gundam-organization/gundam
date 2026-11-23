//
// Created by Nadrino on 23/02/2025.
//

/// @file DialUtils.cpp
/// @brief Implements the DialUtils helpers: knot extraction from ROOT
///        objects, slope filling and the spline shape tests.
/// @ingroup DialDictionary
///
/// The public contract of each function is documented in DialUtils.h. The
/// comments here describe how each function does its job.

#include "DialUtils.h"

#include <Logger.h>

#include <TDirectory.h>


namespace DialUtils{

  /// @par Implementation
  /// Uses ROOT's TObject::InheritsFrom(), so subclasses of TGraph (e.g.
  /// TGraphErrors) are read as graphs. The TGraph test comes first.
  std::vector<DialPoint> getPointList(const TObject *src_){
    if( src_ == nullptr ) return {};

    if( src_->InheritsFrom(TGraph::Class()) ) { return getPointList((TGraph *)src_); }
    if( src_->InheritsFrom(TSpline3::Class()) ) { return getPointList((TSpline3 *)src_); }

    return {};
  }
  /// @par Implementation
  /// A one-point graph is handled directly, because computing slopes with a
  /// one-point TSpline3 crashes ROOT. Otherwise a temporary TSpline3 is built
  /// from the graph and getPointList(const TSpline3*) reads its knots and
  /// derivatives.
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

    TSpline3 spline("", src_); // spline through the graph points, only used to get the slopes
    return getPointList( &spline );
  }
  /// @par Implementation
  /// Reads each knot with TSpline3::GetKnot() and its slope with
  /// TSpline3::Derivative() at the knot position. The first non-finite value
  /// makes the whole list invalid, so an empty list is returned at once.
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
  /// @par Implementation
  /// Copies TGraph::GetX() and TGraph::GetY() point by point, in the order
  /// stored in the graph.
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

  /// @par Implementation
  /// Sets the two end slopes first (one-sided segment slopes) and stops there
  /// for two knots, then loops over the interior knots with getSlope() on
  /// each knot's two neighbours.
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
    size_t idx{0}; // index of the end knot whose slope is being set
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
  /// @par Implementation
  /// Handles the special knots first, returning early for short lists: the
  /// two ends (one-sided segment slopes), then knots 1 and n-2 (Catmull-Rom
  /// slopes), which do not have two segments on each side. The Akima formula
  /// is only applied to knots 2 to n-3, so lists of 5 or more knots.
  /// @note The test `n0 > 1E-6 or n1 < 1E-6` selects the weighted formula
  ///       also when both weights are tiny, which gives NaN when both are
  ///       exactly 0 (see the warning in DialUtils.h).
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

    size_t idx{0}; // index of the special knot whose slope is being set

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
      double mp1 = getSlope(splinePointList_[iPt + 1], splinePointList_[iPt + 2]); // slope of segment iPt+1
      double m0 = getSlope(splinePointList_[iPt], splinePointList_[iPt + 1]); // slope of segment iPt (right of the knot)
      double mm1 = getSlope(splinePointList_[iPt - 1], splinePointList_[iPt]); // slope of segment iPt-1 (left of the knot)
      double mm2 = getSlope(splinePointList_[iPt - 2], splinePointList_[iPt - 1]); // slope of segment iPt-2
      double n1 = std::abs(mp1 - m0); // Akima weight of the left slope mm1
      double n0 = std::abs(mm1 - mm2); // Akima weight of the right slope m0

      if( n0 > 1E-6 or n1 < 1E-6 ) { splinePointList_[iPt].slope = (n1 * mm1 + n0 * m0) / (n1 + n0); } else {
        splinePointList_[iPt].slope = 0.5 * (m0 + mm1);
      }
    }
  }

  /// @par Implementation
  /// Follows the Fritsch-Carlson steps (numbered in the body): segment
  /// slopes `d`, initial tangents `m` (mean of the two neighbouring segment
  /// slopes, one-sided at the ends), zero tangents at sign changes and next
  /// to flat segments, then the ratios `alpha = m[k]/d[k]` and
  /// `beta = m[k+1]/d[k]`. Negative ratios zero the tangent; ratios above 3
  /// limit it to 3 times the segment slope. Finally a positive input slope
  /// above its tangent, or a negative one below it, is replaced by the
  /// tangent; other input slopes are kept.
  /// @note The step comments 4.C and 4.D say "when alpha (beta) is zero";
  ///       the code tests for negative values (`alpha[k] < 0.0`,
  ///       `beta[k] < 0.0`).
  /// @note `d` has one entry per knot, but only the first n-1 are segment
  ///       slopes. The last one stays 0 and is only read where 0 has no
  ///       effect.
  void applyMonotonicCondition(std::vector<DialPoint> &splinePointList_){
    // Apply the Fritsh-Carlson monotonic condition to the slopes.  This
    // adjusts the slopes (when necessary), however, with Catmull-Rom the
    // modified slopes will be ignored and the monotonic criteria is
    // applied as the spline is evaluated (saves memory).
    //
    // F.N.Fritsch, and R.E.Carlson, "Monotone Piecewise Cubic
    // Interpolation" SIAM Journal on Numerical Analysis, Vol. 17, Iss. 2
    // (1980) doi:10.1137/0717021
    //
    // This implementation closely (and pedantically) follows the original
    // paper (see Section 4, page 241). The step comments follow the notes
    // from the Wikipedia Monotonic Cubic Interpolation article. It is
    // slightly adapted from the reference implementation noted in
    // unitTest_ApplyMonotonicCondition.cpp by C.McGrew and relicensed under
    // the GUNDAM licence (LGPL 2.1).

    const std::size_t nPoints{splinePointList_.size()};

    // 1) Find the secant line slopes.
    std::vector<double> d(nPoints); // slope of the segment from knot k to knot k+1
    for (std::size_t k = 0; k<nPoints-1; ++k) {
      d[k] = (splinePointList_[k+1].y-splinePointList_[k].y)/(splinePointList_[k+1].x-splinePointList_[k].x);
    }

    // 2) Initialize tangent for interior data points.
    std::vector<double> m(nPoints); // Fritsch-Carlson tangent at knot k
    for (std::size_t k=1; k<nPoints-1; ++k) {
      m[k] = (d[k-1]+d[k])/2.0;
    }
    m[0] = d[0];
    m[nPoints-1] = d[nPoints-2];

    // 2.A) Set tangents to flat if there is a cusp which is true when
    // ds have opposite signs.
    for (std::size_t k=1; k<nPoints; ++k) {
      if (d[k-1]*d[k] < 0.0) m[k] = 0.0;
    }

    // 3) When the successive points are equal, set the slopes to
    // zero.  This is when d is zero.
    for (std::size_t k = 0; k<nPoints-1; ++k) {
      if (std::abs(d[k]) < std::numeric_limits<double>::epsilon()) {
        m[k] = 0.0;
        m[k+1] = 0.0;
      }
    }

    // 4.A) Calculate alpha
    std::vector<double> alpha(nPoints); // alpha[k] = m[k]/d[k], 0 where d[k] is 0
    for (std::size_t k = 0; k < nPoints; ++k) {
      alpha[k] = 0.0;
      if (std::abs(d[k]) > std::numeric_limits<double>::epsilon()) {
        alpha[k] = m[k]/d[k];
      }
    }

    // 4.B) Calculate beta
    std::vector<double> beta(nPoints); // beta[k] = m[k+1]/d[k], 0 where d[k] is 0
    for (std::size_t k = 0; k < nPoints; ++k) {
      beta[k] = 0.0;
      if (std::abs(d[k]) > std::numeric_limits<double>::epsilon()) {
        if (k < nPoints-1) beta[k] = m[k+1]/d[k];
      }
    }

    // 4.C) When alpha is zero slope is zero
    for (std::size_t k = 0; k < nPoints; ++k) {
      if (alpha[k] < 0.0) m[k] = 0.0;
    }

    // 4.D) When beta is zero next slope is zero.
    for (std::size_t k = 0; k < nPoints-1; ++k) {
      if (beta[k] < 0.0) m[k+1] = 0.0;
    }

    // 5) Prevent overshoot This applies the criteria for set 1 in the
    // Fritzch-Carlson paper (figure 2).
    double bound = 3.0; // largest allowed alpha and beta
    for (std::size_t k = 0; k < nPoints; ++k) {
      if (alpha[k] > bound) m[k] = bound*d[k];
    }
    for (std::size_t k = 0; k < nPoints-1; ++k) {
      if (beta[k] > bound) m[k+1] = bound*d[k];
    }

    // 6) Fix the slopes
    for (std::size_t k = 0; k < nPoints; ++k) {
      double s = splinePointList_[k].slope; // slope after limiting; starts as the input slope
      // Limit the range of the slope
      if (splinePointList_[k].slope > 0.0 and splinePointList_[k].slope > m[k]) s = m[k];
      if (splinePointList_[k].slope < 0.0 and splinePointList_[k].slope < m[k]) s = m[k];
      // If the slope changed, copy it back into the output.
      if (std::abs(s-splinePointList_[k].slope)
          > std::numeric_limits<double>::epsilon()) {
        splinePointList_[k].slope = s;
      }
    }
  }

  /// @par Implementation
  /// Compares every value with the first one and stops at the first value
  /// out of tolerance.
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
  /// @par Implementation
  /// The expected spacing is `(x_last - x_first)/(n - 1)`, so the last knot is
  /// on the lattice by construction and is not tested. Each other knot's
  /// distance from `x_first + i*spacing` is divided by the spacing before
  /// the comparison.
  /// @note If all knots have the same x, the spacing is 0, every test
  ///       compares NaN (0/0), and the function returns true.
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
  /// @par Implementation
  /// Plain division, with no check for equal x.
  double getSlope(const DialPoint &start_, const DialPoint &end_){
    return (end_.y - start_.y) / (end_.x - start_.x);
  }

  /// @par Implementation
  /// Copies the positions and values into two arrays and uses the TSpline3
  /// constructor that takes arrays, with no option and no begin/end
  /// conditions.
  TSpline3 buildTSpline3(const std::vector<DialPoint> &splinePointList_){
    std::vector<double> x; // knot positions
    std::vector<double> y; // knot values

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
