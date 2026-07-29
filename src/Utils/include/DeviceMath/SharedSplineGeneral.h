#ifndef SHARED_SPLINE_GENERAL_H_SEEN
#define SHARED_SPLINE_GENERAL_H_SEEN

#ifndef GUNDAM_DEVICE_MATH_SKIP_COMMON_INCLUDE
#include "DeviceMath/SharedSplineCommon.h"
#endif

namespace GundamDeviceMath {
  // Interpolate one point a spline with non-uniform points.  The spline can
  // have at most 15 knots defined.  With optimization (O1 or more), this
  // about forty times faster than TSpline3.
  //
  // This takes the "index" of the point in the data, the parameter value
  // (x) (that made the index), a minimum (lowerBound) and maximum
  // (upperBound) bound, the buffer of data for this spline, and the number
  // of data elements in the spline data (dim).  The input data is arranged as
  //
  // data[0] -- spline lower bound (not used, kept to match other splines)
  // data[1] -- spline step (not used, kept to match other splines)
  // data[2+3*n+0] -- The function value for knot n (i.e. "Y")
  // data[2+3*n+1] -- The function slope for knot n (i.e. "dYdX")
  // data[2+3*n+2] -- The point for knot n (i.e. "X")
  //
  // There will be "dim" elements in the data[] array.
  //
  // NOTE: CalculateUniformSpline, CalculateGeneralSpline,
  // CalculateCompactSpline, and CalculateMonotonicSpline have very similar,
  // but different calls.  In particular the dim parameter meaning is not
  // consistent.
  GUNDAM_DEVICE_MATH_INLINE
  GUNDAM_DEVICE_MATH_SCALAR EvaluateGeneralSpline(const GUNDAM_DEVICE_MATH_SCALAR x_,
                                                  const GUNDAM_DEVICE_MATH_SCALAR lowerBound_,
                                                  const GUNDAM_DEVICE_MATH_SCALAR upperBound_,
                                                  GUNDAM_DEVICE_MATH_PTR_CONST data_,
                               const int dim_) {

#if defined(CALCULATE_GENERAL_SPLINE_LINEAR_IF)
#warning USING CALCULATE_GENERAL_SPLINE_LINEAR_IF
    // Check to find a point that is less than x.  This is "brute force",
    // but since we know that the splines will usually have 7 or fewer
    // points, this may be faster, or comparable, to a binary search.
    // This is benchmarked at 1.276 ms per call.
    const int knotCount = (dim_ - 2) / 3;
    // PROFILED aspen 2024/05/26 -- 7am (~195 it/sec)
    // LTS/Issue510/AtomicOperationsOnHosts w/ GundamInputsOA2021
    // commit 5e9415e55b258d3d82d562942f03ac07c9937528
    //
    // ==1060642== Profiling result:
    //             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
    //  GPU activities:   38.10%  334.242s    261994  1.2758ms  1.2493ms  1.3344ms  _ZN4hemi6KernelIN78_GLOBAL__N__54_tmpxft_003dc161_00000000_7_WeightGeneralSpline_cpp1_ii_220155e217HEMISplinesKernelEJPdPKdS5_S5_S5_PKiPKsS7_mEEEvT_DpT0_
    int ix = 0;
    if (x_ > data_[2 + 3 * (ix + 1) + 2] && ix < knotCount - 2) ++ix; // 1
    if (x_ > data_[2 + 3 * (ix + 1) + 2] && ix < knotCount - 2) ++ix; // 2
    if (x_ > data_[2 + 3 * (ix + 1) + 2] && ix < knotCount - 2) ++ix; // 3
    if (x_ > data_[2 + 3 * (ix + 1) + 2] && ix < knotCount - 2) ++ix; // 4
    if (x_ > data_[2 + 3 * (ix + 1) + 2] && ix < knotCount - 2) ++ix; // 5
    if (x_ > data_[2 + 3 * (ix + 1) + 2] && ix < knotCount - 2) ++ix; // 6
    if (x_ > data_[2 + 3 * (ix + 1) + 2] && ix < knotCount - 2) ++ix; // 7
    if (x_ > data_[2 + 3 * (ix + 1) + 2] && ix < knotCount - 2) ++ix; // 8
    if (x_ > data_[2 + 3 * (ix + 1) + 2] && ix < knotCount - 2) ++ix; // 9
    if (x_ > data_[2 + 3 * (ix + 1) + 2] && ix < knotCount - 2) ++ix; // 10
    if (x_ > data_[2 + 3 * (ix + 1) + 2] && ix < knotCount - 2) ++ix; // 11
    if (x_ > data_[2 + 3 * (ix + 1) + 2] && ix < knotCount - 2) ++ix; // 12
    if (x_ > data_[2 + 3 * (ix + 1) + 2] && ix < knotCount - 2) ++ix; // 13
    if (x_ > data_[2 + 3 * (ix + 1) + 2] && ix < knotCount - 2) ++ix; // 14
    if (x_ > data_[2 + 3 * (ix + 1) + 2] && ix < knotCount - 2) ++ix; // 15
#elif defined(CALCULATE_GENERAL_SPLINE_LINEAR_MULT)
#warning USING CALCULATE_GENERAL_SPLINE_LINEAR_MULT
    // Check to find a point that is less than x.  This is "brute force",
    // but since we know that the splines will usually have 7 or fewer
    // points, this may be faster, or comparable, to a binary search.
    // This does the calculation without ifs (which can be slower for
    // SIMD). This is benchmarked at 1.275 ms per call.
    //
    // PROFILED aspen 2024/05/26 -- 7:30am (~180 it/sec)
    // LTS/Issue510/AtomicOperationsOnHosts w/ GundamInputsOA2021
    // commit 5e9415e55b258d3d82d562942f03ac07c9937528
    //
    // ==1062228== Profiling result:
    //             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
    //  GPU activities:   38.08%  334.068s    262010  1.2750ms  1.2527ms  1.3348ms  _ZN4hemi6KernelIN78_GLOBAL__N__54_tmpxft_001031e3_00000000_7_WeightGeneralSpline_cpp1_ii_220155e217HEMISplinesKernelEJPdPKdS5_S5_S5_PKiPKsS7_mEEEvT_DpT0_
    const int knotCount = (dim_ - 2) / 3 - 2;
    int ix = 0;
    ix += (x_ > data_[2 + 3 * (ix + 1) + 2]) * (ix < knotCount); // 1
    ix += (x_ > data_[2 + 3 * (ix + 1) + 2]) * (ix < knotCount); // 2
    ix += (x_ > data_[2 + 3 * (ix + 1) + 2]) * (ix < knotCount); // 3
    ix += (x_ > data_[2 + 3 * (ix + 1) + 2]) * (ix < knotCount); // 4
    ix += (x_ > data_[2 + 3 * (ix + 1) + 2]) * (ix < knotCount); // 5
    ix += (x_ > data_[2 + 3 * (ix + 1) + 2]) * (ix < knotCount); // 6
    ix += (x_ > data_[2 + 3 * (ix + 1) + 2]) * (ix < knotCount); // 7
    ix += (x_ > data_[2 + 3 * (ix + 1) + 2]) * (ix < knotCount); // 8
    ix += (x_ > data_[2 + 3 * (ix + 1) + 2]) * (ix < knotCount); // 9
    ix += (x_ > data_[2 + 3 * (ix + 1) + 2]) * (ix < knotCount); // 10
    ix += (x_ > data_[2 + 3 * (ix + 1) + 2]) * (ix < knotCount); // 11
    ix += (x_ > data_[2 + 3 * (ix + 1) + 2]) * (ix < knotCount); // 12
    ix += (x_ > data_[2 + 3 * (ix + 1) + 2]) * (ix < knotCount); // 13
    ix += (x_ > data_[2 + 3 * (ix + 1) + 2]) * (ix < knotCount); // 14
    ix += (x_ > data_[2 + 3 * (ix + 1) + 2]) * (ix < knotCount); // 15
#elif defined(CALCULATE_GENERAL_SPLINE_LOOPED_CHECK)
    const int knotCount = (dim_ - 2) / 3 - 2;
    int ix = 0;
#define CHECK_OFFSET(ioff)  if ((ix + ioff < knotCount) && (x_ > data_[2 + 3 * (ix + ioff) + 2])) ix += ioff
    for( int offset = 1 << (31 - __builtin_clz(knotCount)) ; offset > 0 ; offset >>= 1 ){
      CHECK_OFFSET(offset);
    }
#undef CHECK_OFFSET
#else /* CALCULATE_GENERAL_SPLINE_BINARY_IF */
    // Check to find a point that is less than x.  This is a "brute force"
    // binary search, and the "if" has been checked and is efficient with
    // CUDA.  This is benchmarked at 0.526 ms per call.
    //
    // PROFILED aspen 2024/05/26 -- 8:00am (~190 it/sec)
    // LTS/Issue510/AtomicOperationsOnHosts w/ GundamInputsOA2021
    // commit 5e9415e55b258d3d82d562942f03ac07c9937528
    //
    // ==1063988== Profiling result:
    //             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
    //                    20.26%  137.925s    262004  526.42us  517.37us  550.90us  _ZN4hemi6KernelIN78_GLOBAL__N__54_tmpxft_00103aad_00000000_7_WeightGeneralSpline_cpp1_ii_220155e217HEMISplinesKernelEJPdPKdS5_S5_S5_PKiPKsS7_mEEEvT_DpT0_
    //
    const int knotCount = (dim_ - 2) / 3 - 1;
    int ix = 0;
#define CHECK_OFFSET(ioff)  if ((ix + ioff < knotCount) && (x_ > data_[2 + 3 * (ix + ioff) + 2])) ix += ioff
    CHECK_OFFSET(16);
    CHECK_OFFSET(8);
    CHECK_OFFSET(4);
    CHECK_OFFSET(2);
    CHECK_OFFSET(1);
#undef CHECK_OFFSET
#endif

    const GUNDAM_DEVICE_MATH_SCALAR x1 = data_[2 + 3 * ix + 2];
    const GUNDAM_DEVICE_MATH_SCALAR x2 = data_[2 + 3 * (ix + 1) + 2];
    const GUNDAM_DEVICE_MATH_SCALAR step = x2 - x1;
    const GUNDAM_DEVICE_MATH_SCALAR fx = (x_ - x1) / step;

    const GUNDAM_DEVICE_MATH_SCALAR p1 = data_[2 + 3 * ix];
    const GUNDAM_DEVICE_MATH_SCALAR m1 = data_[2 + 3 * ix + 1] * step;
    const GUNDAM_DEVICE_MATH_SCALAR p2 = data_[2 + 3 * (ix + 1)];
    const GUNDAM_DEVICE_MATH_SCALAR m2 = data_[2 + 3 * (ix + 1) + 1] * step;

#ifdef DO_NOT_USE_HORNER_FACTORIZATION
    const GUNDAM_DEVICE_MATH_SCALAR fxx = fx * fx;
    const GUNDAM_DEVICE_MATH_SCALAR fxxx = fx * fxx;
    const GUNDAM_DEVICE_MATH_SCALAR t = GUNDAM_DEVICE_MATH_SCALAR(3.0) * fxx
                                        - GUNDAM_DEVICE_MATH_SCALAR(2.0) * fxxx;
    GUNDAM_DEVICE_MATH_SCALAR value = p1 - p1 * t
                                      + m1 * (fxxx - GUNDAM_DEVICE_MATH_SCALAR(2.0) * fxx + fx)
                                      + p2 * t + m2 * (fxxx - fxx);
#else
    const GUNDAM_DEVICE_MATH_SCALAR value = ((((GUNDAM_DEVICE_MATH_SCALAR(2.0) * p1
                                                    - GUNDAM_DEVICE_MATH_SCALAR(2.0) * p2 + m2 + m1) * fx
                                               + GUNDAM_DEVICE_MATH_SCALAR(3.0) * p2
                                               - GUNDAM_DEVICE_MATH_SCALAR(3.0) * p1
                                               - m2 - GUNDAM_DEVICE_MATH_SCALAR(2.0) * m1) * fx
                           + m1) * fx
                          + p1);
#endif

    return ClampResponse(value, lowerBound_, upperBound_);
  }
}

#endif
