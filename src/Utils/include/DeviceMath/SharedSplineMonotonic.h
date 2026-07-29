#ifndef SHARED_SPLINE_MONOTONIC_H_SEEN
#define SHARED_SPLINE_MONOTONIC_H_SEEN

#include "DeviceMath/SharedSplineCommon.h"

namespace GundamDeviceMath {
  // Interpolate one point using a monotonic spline.  This takes the "index"
  // of the point in the data, the parameter value (that made the index), a
  // minimum and maximum bound, the buffer of data for this spline, and the
  // number of knots in the spline data.  The input data is arranged as:
  //
  // data[0] -- spline lower bound
  // data[1] -- spline step between X values
  // data[2+n+0] -- The function value for knot n (0 to dim-1)
  //
  // NOTE: CalculateUniformSpline, CalculateGeneralSpline,
  // CalculateCompactSpline, and CalculateMonotonicSpline have very similar,
  // but different calls.  In particular the dim parameter meaning is not
  // consistent.
  GUNDAM_DEVICE_MATH_INLINE
  double EvaluateMonotonicSpline(const double x_,
                                 const double lowerBound_,
                                 const double upperBound_,
                                 const GUNDAM_DEVICE_MATH_FLOAT* data_,
                                 const int dim_) {
    const double low = data_[0];
    const double step = data_[1];
    const double xx = (x_ - low) / step;
    const int ix = (xx < 0) ? int(xx) - 1 : int(xx);

    // Interpolate between p2 and p3
    // ix-2 ix-1 ix   ix+1 ix+2 ix+3
    // p0   p1   p2---p3   p4   p5
    //   d10  d21  d32  d43  d54
    // m0| |m1| |m2| |m3| |m4| |m5
    //  a0 ||a1 ||a2 ||a3 ||a4 ||a5
    //     b0   b1   b2   b3   b4
    const int d21_0 = ClampSplineSegmentIndex(ix - 1, dim_);
    const int d21_1 = d21_0 + 1;
    const int d32_0 = ClampSplineSegmentIndex(ix, dim_);
    const int d32_1 = d32_0 + 1;
    const int d43_0 = ClampSplineSegmentIndex(ix + 1, dim_);
    const int d43_1 = d43_0 + 1;

    const double p2 = data_[2 + d32_0];
    const double p3 = data_[2 + d32_1];
    const double fx = xx - d32_0;
    const double d21 = data_[2 + d21_1] - data_[2 + d21_0];
    const double d32 = p3 - p2;
    const double d43 = data_[2 + d43_1] - data_[2 + d43_0];

    double m2 = 0.5 * (d21 + d32);
    double m3 = 0.5 * (d32 + d43);

    // Apply the Fritsch-Carlson monotonic condition to the slopes.
    //
    // F.N. Fritsch and R.E. Carlson, "Monotone Piecewise Cubic
    // Interpolation", SIAM Journal on Numerical Analysis, Vol. 17, Iss. 2
    // (1980) doi:10.1137/0717021
    //
    // Deal with cusp points and flat areas.
    if( d32 * d21 <= 0.0 ){ m2 = 0.0; }
    if( d43 * d32 <= 0.0 ){ m3 = 0.0; }

    const double ad21 = Abs(d21);
    const double ad32 = Abs(d32);
    const double ad43 = Abs(d43);
    const double delta2 = 3.0 * Min(ad21, ad32);
    const double delta3 = 3.0 * Min(ad32, ad43);

    if( m2 > delta2 ){ m2 = delta2; }
    if( m2 < -delta2 ){ m2 = -delta2; }
    if( m3 > delta3 ){ m3 = delta3; }
    if( m3 < -delta3 ){ m3 = -delta3; }

    // Cubic spline with the points and slopes, factored via Horner's method.
    const double value = ((((2.0 * p2 - 2.0 * p3 + m3 + m2) * fx
                            + 3.0 * p3 - 3.0 * p2 - m3 - 2.0 * m2) * fx
                           + m2) * fx
                          + p2);

    return ClampResponse(value, lowerBound_, upperBound_);
  }
}

#endif
