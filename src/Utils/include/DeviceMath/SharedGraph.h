#ifndef SHARED_GRAPH_H_SEEN
#define SHARED_GRAPH_H_SEEN

#ifndef GUNDAM_DEVICE_MATH_SKIP_COMMON_INCLUDE
#include "DeviceMath/SharedSplineCommon.h"
#endif

namespace GundamDeviceMath {
  /// Interpolate one point in a graph with non-uniform points.  The graph can
  /// have at most 15 knots defined.
  ///
  /// This takes the parameter value, a minimum and maximum bound, the
  /// buffer of data for this graph, and the number of data elements in the
  /// graph data.  The input data is arranged as
  ///
  /// data[2*n+0] -- The function value for knot n
  /// data[2*n+1] -- The point for knot n
  ///
  /// NOTE: CalculateUniformSpline, CalculateGeneralSpline,
  /// CalculateCompactSpline, and CalculateMonotonicSpline have very similar,
  /// but different calls.  In particular the dim parameter meaning is not
  /// consistent.
  GUNDAM_DEVICE_MATH_INLINE
  GUNDAM_DEVICE_MATH_SCALAR EvaluateGraph(const GUNDAM_DEVICE_MATH_SCALAR x_,
                                          const GUNDAM_DEVICE_MATH_SCALAR lowerBound_,
                                          const GUNDAM_DEVICE_MATH_SCALAR upperBound_,
                                          GUNDAM_DEVICE_MATH_PTR_CONST data_,
                       const int dim_) {

    if( dim_ < 4 ){ return data_[0]; }

    const int knotCount = dim_ / 2;
    int ix = 0;
#define CHECK_OFFSET(ioff)  if ((ix + ioff < knotCount) && (x_ > data_[2 * (ix + ioff) + 1])) ix += ioff
    for( int offset = 1 << (31 - __builtin_clz(knotCount)) ; offset > 0 ; offset >>= 1 ){
      CHECK_OFFSET(offset);
    }
#undef CHECK_OFFSET

    if( ix + 1 >= knotCount ){ ix--; }

    const GUNDAM_DEVICE_MATH_SCALAR p1 = data_[2 * ix];
    const GUNDAM_DEVICE_MATH_SCALAR x1 = data_[2 * ix + 1];
    const GUNDAM_DEVICE_MATH_SCALAR p2 = data_[2 * (ix + 1)];
    const GUNDAM_DEVICE_MATH_SCALAR x2 = data_[2 * (ix + 1) + 1];

    const GUNDAM_DEVICE_MATH_SCALAR step = x2 - x1;
    const GUNDAM_DEVICE_MATH_SCALAR fx = (x_ - x1) / step;
    const GUNDAM_DEVICE_MATH_SCALAR value = p1 + fx * (p2 - p1);

    return ClampResponse(value, lowerBound_, upperBound_);
  }
}

#endif
