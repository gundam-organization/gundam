//
// Created by Nadrino on 23/02/2025.
//

#include "SplineUtils.h"

#include <GundamUtils.h>
#include <Logger.h>
#include <TDirectory.h>


namespace SplineUtils{
  std::vector<SplinePoint> getSplinePointList(const TObject *src_){
    if( src_ == nullptr ) return {};

    if( src_->InheritsFrom(TGraph::Class()) ) { return getSplinePointList((TGraph *)src_); }
    if( src_->InheritsFrom(TSpline3::Class()) ) { return getSplinePointList((TSpline3 *)src_); }

    return {};
  }
  std::vector<SplinePoint> getSplinePointList(const TGraph *src_){
    if( src_ == nullptr ) return {};

    std::vector<SplinePoint> out;
    out.reserve(src_->GetN());

    for( int iPt = 0; iPt < src_->GetN(); iPt++ ) {
      out.emplace_back();
      auto &point = out.back();

      point.x = src_->GetX()[iPt];
      point.y = src_->GetY()[iPt];
    }

    fillTSpline3Slopes(out);

    return out;
  }
  std::vector<SplinePoint> getSplinePointList(const TSpline3 *src_){
    if( src_ == nullptr ) return {};
    if( src_->GetNp() == 0 ) return {};

    std::vector<SplinePoint> out;
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

  void fillTSpline3Slopes(std::vector<SplinePoint>& splinePointList_, const std::string& opt_){
    struct TestSpline{
      Int_t fNp{0};
      Int_t fBegCond = -1;
      Int_t fEndCond = -1;
      Double_t fValBeg = 0.;
      Double_t fValEnd = 0.;

      struct TestSplinePoly3{
        Double_t fX = 0.;     ///< Abscissa
        Double_t fY = 0.;     ///< Constant term
        Double_t fB = 0.; ///< First order expansion coefficient :  fB*1! is the first derivative at x
        Double_t fC = 0.; ///< Second order expansion coefficient : fC*2! is the second derivative at x
        Double_t fD = 0.; ///< Third order expansion coefficient :  fD*3! is the third derivative at x

        Double_t &X() { return fX; }
        Double_t &Y() { return fY; }
        Double_t &B() { return fB; }
        Double_t &C() { return fC; }
        Double_t &D() { return fD; }
        Double_t Derivative(Double_t x) const{
          Double_t dx = x - fX;
          return (fB + dx * (2 * fC + 3 * fD * dx));
        }
      };
      std::vector<TestSplinePoly3> fPoly;

      void BuildCoeff(){
        Int_t i, j, l, m;
        Double_t divdf1, divdf3, dtau, g = 0;
        //***** a tridiagonal linear system for the unknown slopes s(i) of
        //  f  at tau(i), i=1,...,n, is generated and then solved by gauss elim-
        //  ination, with s(i) ending up in c(2,i), all i.
        //     c(3,.) and c(4,.) are used initially for temporary storage.
        l = fNp - 1;
        // compute first differences of x sequence and store in C also,
        // compute first divided difference of data and store in D.
        for( m = 1; m < fNp; ++m ) {
          fPoly[m].C() = fPoly[m].X() - fPoly[m - 1].X();
          fPoly[m].D() = (fPoly[m].Y() - fPoly[m - 1].Y()) / fPoly[m].C();
        }
        // construct first equation from the boundary condition, of the form
        //             D[0]*s[0] + C[0]*s[1] = B[0]
        if( fBegCond == 0 ) {
          if( fNp == 2 ) {
            //     no condition at left end and n = 2.
            fPoly[0].D() = 1.;
            fPoly[0].C() = 1.;
            fPoly[0].B() = 2. * fPoly[1].D();
          } else {
            //     not-a-knot condition at left end and n .gt. 2.
            fPoly[0].D() = fPoly[2].C();
            fPoly[0].C() = fPoly[1].C() + fPoly[2].C();
            fPoly[0].B() = ((fPoly[1].C() + 2. * fPoly[0].C()) * fPoly[1].D() * fPoly[2].C() + fPoly[1].C() * fPoly[1].
                            C() * fPoly[2].D()) / fPoly[0].C();
          }
        } else if( fBegCond == 1 ) {
          //     slope prescribed at left end.
          fPoly[0].B() = fValBeg;
          fPoly[0].D() = 1.;
          fPoly[0].C() = 0.;
        } else if( fBegCond == 2 ) {
          //     second derivative prescribed at left end.
          fPoly[0].D() = 2.;
          fPoly[0].C() = 1.;
          fPoly[0].B() = 3. * fPoly[1].D() - fPoly[1].C() / 2. * fValBeg;
        }
        if( fNp > 2 ) {
          //  if there are interior knots, generate the corresp. equations and car-
          //  ry out the forward pass of gauss elimination, after which the m-th
          //  equation reads    D[m]*s[m] + C[m]*s[m+1] = B[m].
          for( m = 1; m < l; ++m ) {
            g = -fPoly[m + 1].C() / fPoly[m - 1].D();
            fPoly[m].B() = g * fPoly[m - 1].B() + 3. * (
                             fPoly[m].C() * fPoly[m + 1].D() + fPoly[m + 1].C() * fPoly[m].D());
            fPoly[m].D() = g * fPoly[m - 1].C() + 2. * (fPoly[m].C() + fPoly[m + 1].C());
          }
          // construct last equation from the second boundary condition, of the form
          //           (-g*D[n-2])*s[n-2] + D[n-1]*s[n-1] = B[n-1]
          //     if slope is prescribed at right end, one can go directly to back-
          //     substitution, since c array happens to be set up just right for it
          //     at this point.
          if( fEndCond == 0 ) {
            if( fNp > 3 || fBegCond != 0 ) {
              //     not-a-knot and n .ge. 3, and either n.gt.3 or  also not-a-knot at
              //     left end point.
              g = fPoly[fNp - 2].C() + fPoly[fNp - 1].C();
              fPoly[fNp - 1].B() = ((fPoly[fNp - 1].C() + 2. * g) * fPoly[fNp - 1].D() * fPoly[fNp - 2].C()
                                    + fPoly[fNp - 1].C() * fPoly[fNp - 1].C() * (
                                      fPoly[fNp - 2].Y() - fPoly[fNp - 3].Y()) / fPoly[fNp - 2].C()) / g;
              g = -g / fPoly[fNp - 2].D();
              fPoly[fNp - 1].D() = fPoly[fNp - 2].C();
            } else {
              //     either (n=3 and not-a-knot also at left) or (n=2 and not not-a-
              //     knot at left end point).
              fPoly[fNp - 1].B() = 2. * fPoly[fNp - 1].D();
              fPoly[fNp - 1].D() = 1.;
              g = -1. / fPoly[fNp - 2].D();
            }
          } else if( fEndCond == 1 ) {
            fPoly[fNp - 1].B() = fValEnd;
            goto L30;
          } else if( fEndCond == 2 ) {
            //     second derivative prescribed at right endpoint.
            fPoly[fNp - 1].B() = 3. * fPoly[fNp - 1].D() + fPoly[fNp - 1].C() / 2. * fValEnd;
            fPoly[fNp - 1].D() = 2.;
            g = -1. / fPoly[fNp - 2].D();
          }
        } else {
          if( fEndCond == 0 ) {
            if( fBegCond > 0 ) {
              //     either (n=3 and not-a-knot also at left) or (n=2 and not not-a-
              //     knot at left end point).
              fPoly[fNp - 1].B() = 2. * fPoly[fNp - 1].D();
              fPoly[fNp - 1].D() = 1.;
              g = -1. / fPoly[fNp - 2].D();
            } else {
              //     not-a-knot at right endpoint and at left endpoint and n = 2.
              fPoly[fNp - 1].B() = fPoly[fNp - 1].D();
              goto L30;
            }
          } else if( fEndCond == 1 ) {
            fPoly[fNp - 1].B() = fValEnd;
            goto L30;
          } else if( fEndCond == 2 ) {
            //     second derivative prescribed at right endpoint.
            fPoly[fNp - 1].B() = 3. * fPoly[fNp - 1].D() + fPoly[fNp - 1].C() / 2. * fValEnd;
            fPoly[fNp - 1].D() = 2.;
            g = -1. / fPoly[fNp - 2].D();
          }
        }
        // complete forward pass of gauss elimination.
        LogThrowIf(fPoly.size() != fNp);
        fPoly[fNp - 1].D() = g * fPoly[fNp - 2].C() + fPoly[fNp - 1].D();
        fPoly[fNp - 1].B() = (g * fPoly[fNp - 2].B() + fPoly[fNp - 1].B()) / fPoly[fNp - 1].D();
        // carry out back substitution
      L30:
        j = l - 1;
        do {
          fPoly[j].B() = (fPoly[j].B() - fPoly[j].C() * fPoly[j + 1].B()) / fPoly[j].D();
          --j;
        } while( j >= 0 );
        //****** generate cubic coefficients in each interval, i.e., the deriv.s
        //  at its left endpoint, from value and slope at its endpoints.
        for( i = 1; i < fNp; ++i ) {
          dtau = fPoly[i].C();
          divdf1 = (fPoly[i].Y() - fPoly[i - 1].Y()) / dtau;
          divdf3 = fPoly[i - 1].B() + fPoly[i].B() - 2. * divdf1;
          fPoly[i - 1].C() = (divdf1 - fPoly[i - 1].B() - divdf3) / dtau;
          fPoly[i - 1].D() = (divdf3 / dtau) / dtau;
        }
      }
      void SetCond(const char *opt){
        const char *b1 = strstr(opt,"b1");
        const char *e1 = strstr(opt,"e1");
        const char *b2 = strstr(opt,"b2");
        const char *e2 = strstr(opt,"e2");
        LogThrowIf(b1 && b2, "Cannot specify first and second derivative at first point");
        LogThrowIf(e1 && e2, "Cannot specify first and second derivative at last point");
        if (b1) fBegCond=1;
        else if (b2) fBegCond=2;
        if (e1) fEndCond=1;
        else if (e2) fEndCond=2;
      }
    };

    // Create the polynomial terms and fill
    // them with node information
    TestSpline s;
    s.SetCond(opt_.c_str());
    s.fNp = static_cast<int>(splinePointList_.size());
    s.fPoly.resize(s.fNp);
    for( Int_t i=0 ; i<s.fNp ; ++i ){
      s.fPoly[i].X() = splinePointList_[i].x;
      s.fPoly[i].Y() = splinePointList_[i].y;
    }
    s.BuildCoeff();

    for( Int_t i=0 ; i<s.fNp ; ++i ) {
      splinePointList_[i].slope = s.fPoly[i].Derivative(splinePointList_[i].x);
    }

  }
  void fillCatmullRomSlopes(std::vector<SplinePoint> &splinePointList_){
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
  void fillAkimaSlopes(std::vector<SplinePoint> &splinePointList_){
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
  void applyMonotonicCondition(std::vector<SplinePoint> &splinePointList_){
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

  bool isFlat(const std::vector<SplinePoint> &splinePointList_, double tolerance_){
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
  bool isUniform(const std::vector<SplinePoint> &splinePointList_, double tolerance_){
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
  double getSlope(const SplinePoint &start_, const SplinePoint &end_){
    return (end_.y - start_.y) / (end_.x - start_.x);
  }

  TSpline3 buildTSpline3(const std::vector<SplinePoint> &splinePointList_){
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
