#ifndef SHARED_SPLINE_UNIFORM_H_SEEN
#define SHARED_SPLINE_UNIFORM_H_SEEN

#include "DeviceMath/SharedSplineCommon.h"

namespace GundamDeviceMath {
  // Interpolate one point using a spline with uniformly spaced knots.  This
  // is much faster than TSpline3.  This takes the "index" of the point in the
  // data, the parameter value (that made the index), a minimum and maximum
  // output value, the buffer of data for this spline, and the number of data
  // elements in the spline data.  The input data is arranged as:
  //
  // data[0] -- spline lower bound
  // data[1] -- spline step
  // data[2+2*n+0] -- The function value for knot n (0 to dim-3)
  // data[2+2*n+1] -- The function slope for knot n
  //
  // NOTE: CalculateUniformSpline, CalculateGeneralSpline,
  // CalculateCompactSpline, and CalculateMonotonicSpline have very similar,
  // but different calls.  In particular the dim parameter meaning is not
  // consistent.
  GUNDAM_DEVICE_MATH_INLINE
  double EvaluateUniformSpline(const double x_,
                               const double lowerBound_,
                               const double upperBound_,
                               const GUNDAM_DEVICE_MATH_FLOAT* data_,
                               const int dim_) {
    const double step = data_[1];
    const double xx = (x_ - data_[0]) / step;
    int ix = int(xx);
    if( ix < 0 ){ ix = 0; }
    if( 2 * ix + 7 > dim_ ){ ix = (dim_ - 2) / 2 - 2; }

    const double fx = xx - ix;
    const double p1 = data_[2 + 2 * ix];
    const double m1 = data_[2 + 2 * ix + 1] * step;
    const double p2 = data_[2 + 2 * ix + 2];
    const double m2 = data_[2 + 2 * ix + 3] * step;

    // Cubic spline with the points and slopes, factored via Horner's method.
    const double value = ((((2.0 * p1 - 2.0 * p2 + m2 + m1) * fx
                            + 3.0 * p2 - 3.0 * p1 - m2 - 2.0 * m1) * fx
                           + m1) * fx
                          + p1);

    return ClampResponse(value, lowerBound_, upperBound_);
  }
}

#endif
