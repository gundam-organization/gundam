#ifndef SHARED_SPLINE_COMPACT_H_SEEN
#define SHARED_SPLINE_COMPACT_H_SEEN

#include "DeviceMath/SharedSplineCommon.h"

namespace GundamDeviceMath {
  // Interpolate one point using a compact spline.  This takes the "index"
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
  double EvaluateCompactSpline(const double x_,
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
    const double m2 = 0.5 * (d21 + d32);
    const double m3 = 0.5 * (d32 + d43);

    // Cubic spline with the points and slopes, factored via Horner's method.
    const double value = ((((2.0 * p2 - 2.0 * p3 + m3 + m2) * fx
                            + 3.0 * p3 - 3.0 * p2 - m3 - 2.0 * m2) * fx
                           + m2) * fx
                          + p2);

    return ClampResponse(value, lowerBound_, upperBound_);
  }
}

#endif
