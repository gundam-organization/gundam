#ifndef SHARED_SPLINE_UNIFORM_H_SEEN
#define SHARED_SPLINE_UNIFORM_H_SEEN

#ifndef GUNDAM_DEVICE_MATH_SKIP_COMMON_INCLUDE
#include "DeviceMath/SharedSplineCommon.h"
#endif

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
  GUNDAM_DEVICE_MATH_SCALAR EvaluateUniformSpline(const GUNDAM_DEVICE_MATH_SCALAR x_,
                                                  const GUNDAM_DEVICE_MATH_SCALAR lowerBound_,
                                                  const GUNDAM_DEVICE_MATH_SCALAR upperBound_,
                                                  GUNDAM_DEVICE_MATH_PTR_CONST data_,
                               const int dim_) {
    const GUNDAM_DEVICE_MATH_SCALAR step = data_[1];
    const GUNDAM_DEVICE_MATH_SCALAR xx = (x_ - data_[0]) / step;
    int ix = int(xx);
    if( ix < 0 ){ ix = 0; }
    if( 2 * ix + 7 > dim_ ){ ix = (dim_ - 2) / 2 - 2; }

    const GUNDAM_DEVICE_MATH_SCALAR fx = xx - ix;
    const GUNDAM_DEVICE_MATH_SCALAR p1 = data_[2 + 2 * ix];
    const GUNDAM_DEVICE_MATH_SCALAR m1 = data_[2 + 2 * ix + 1] * step;
    const GUNDAM_DEVICE_MATH_SCALAR p2 = data_[2 + 2 * ix + 2];
    const GUNDAM_DEVICE_MATH_SCALAR m2 = data_[2 + 2 * ix + 3] * step;

    // Cubic spline with the points and slopes, factored via Horner's method.
    const GUNDAM_DEVICE_MATH_SCALAR value = ((((GUNDAM_DEVICE_MATH_SCALAR(2.0) * p1
                                                    - GUNDAM_DEVICE_MATH_SCALAR(2.0) * p2 + m2 + m1) * fx
                                               + GUNDAM_DEVICE_MATH_SCALAR(3.0) * p2
                                               - GUNDAM_DEVICE_MATH_SCALAR(3.0) * p1
                                               - m2 - GUNDAM_DEVICE_MATH_SCALAR(2.0) * m1) * fx
                           + m1) * fx
                          + p1);

    return ClampResponse(value, lowerBound_, upperBound_);
  }
}

#endif
