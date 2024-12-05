#ifndef CALCULATE_GENERAL_SPLINE_H_SEEN
// Calculate a spline specified by the position, value, and slope at each
// knot.  The knot positions must be in increasing order.  This adds a local
// function that can be called from CPU (with c++), or a GPU (with CUDA).
// With G++ running with "O1", this is about forty times faster than TSpline3.

// Wrap the CUDA compiler attributes into a definition.  When this is compiled
// with a CUDA compiler __CUDACC__ will be defined.  In that case, the code
// will be compiled with cuda attributes for both the host (i.e. __host__) and
// gpu (i.e. __device__).  If it's compiled with a normal C compiler, this is
// compiled as inline.
#ifndef DEVICE_CALLABLE_INLINE
#ifdef __CUDACC__
// This is used with a cuda compiler (i.e. nvcc)
#define DEVICE_CALLABLE_INLINE __host__ __device__ inline
#else
// This is used for a non-cuda compiler
#define DEVICE_CALLABLE_INLINE /* __host__ __device__ inline */
#endif
#endif

// Allow the floating point type to be overriden.  This would normally be done
// using a typedef, but that doesn't play well with the CUDA compiler.
#ifndef DEVICE_FLOATING_POINT
#define DEVICE_FLOATING_POINT double
#endif

// Place in a private name space so it plays nicely with CUDA
namespace {
    // Interpolate one point a spline with non-uniform points.  The spline can
    // have at most 15 knots defined.  With optimization (O1 or more), this
    // about forty times faster than TSpline3.
    //
    // This takes the "index" of the point in the data, the parameter value
    // (x) (that made the index), a minimum (lowerBound) and maximum
    // (upperBound) bound, the buffer of data for this spline, and the number
    // of data elements in the spline data (dim).  The input data is arrange as
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
    DEVICE_CALLABLE_INLINE
    double CalculateGeneralSpline(const double x,
                                  const double lowerBound, double upperBound,
                                  const DEVICE_FLOATING_POINT* data,
                                  const int dim) {

#if defined(CALCULATE_GENERAL_SPLINE_LINEAR_IF)
#warning USING CALCULATE_GENERAL_SPLINE_LINEAR_IF
        // Check to find a point that is less than x.  This is "brute force",
        // but since we know that the splines will usually have 7 or fewer
        // points, this may be faster, or comparable, to a binary search.
        const int knotCount = (dim-2)/3;
// PROFILED aspen 2024/05/26 -- 7am (~195 it/sec)
// LTS/Issue510/AtomicOperationsOnHosts w/ GundamInputsOA2021
// commit 5e9415e55b258d3d82d562942f03ac07c9937528
//
// ==1060642== Profiling result:
//             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//  GPU activities:   38.10%  334.242s    261994  1.2758ms  1.2493ms  1.3344ms  _ZN4hemi6KernelIN78_GLOBAL__N__54_tmpxft_003dc161_00000000_7_WeightGeneralSpline_cpp1_ii_220155e217HEMISplinesKernelEJPdPKdS5_S5_S5_PKiPKsS7_mEEEvT_DpT0_
        int ix = 0;
        if (x > data[2+3*(ix+1)+2] && ix < knotCount-2) ++ix; // 1
        if (x > data[2+3*(ix+1)+2] && ix < knotCount-2) ++ix; // 2
        if (x > data[2+3*(ix+1)+2] && ix < knotCount-2) ++ix; // 3
        if (x > data[2+3*(ix+1)+2] && ix < knotCount-2) ++ix; // 4
        if (x > data[2+3*(ix+1)+2] && ix < knotCount-2) ++ix; // 5
        if (x > data[2+3*(ix+1)+2] && ix < knotCount-2) ++ix; // 6
        if (x > data[2+3*(ix+1)+2] && ix < knotCount-2) ++ix; // 7
        if (x > data[2+3*(ix+1)+2] && ix < knotCount-2) ++ix; // 8
        if (x > data[2+3*(ix+1)+2] && ix < knotCount-2) ++ix; // 9
        if (x > data[2+3*(ix+1)+2] && ix < knotCount-2) ++ix; // 10
        if (x > data[2+3*(ix+1)+2] && ix < knotCount-2) ++ix; // 11
        if (x > data[2+3*(ix+1)+2] && ix < knotCount-2) ++ix; // 12
        if (x > data[2+3*(ix+1)+2] && ix < knotCount-2) ++ix; // 13
        if (x > data[2+3*(ix+1)+2] && ix < knotCount-2) ++ix; // 14
        if (x > data[2+3*(ix+1)+2] && ix < knotCount-2) ++ix; // 15
#elif defined(CALCULATE_GENERAL_SPLINE_LINEAR_MULT)
#warning USING CALCULATE_GENERAL_SPLINE_LINEAR_MULT
        // Check to find a point that is less than x.  This is "brute force",
        // but since we know that the splines will usually have 7 or fewer
        // points, this may be faster, or comparable, to a binary search.
        // This does the calculation without ifs (which can be slower for
        // SIMD).
// PROFILED aspen 2024/05/26 -- 7:30am (~180 it/sec)
// LTS/Issue510/AtomicOperationsOnHosts w/ GundamInputsOA2021
// commit 5e9415e55b258d3d82d562942f03ac07c9937528
//
// ==1062228== Profiling result:
//             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//  GPU activities:   38.08%  334.068s    262010  1.2750ms  1.2527ms  1.3348ms  _ZN4hemi6KernelIN78_GLOBAL__N__54_tmpxft_001031e3_00000000_7_WeightGeneralSpline_cpp1_ii_220155e217HEMISplinesKernelEJPdPKdS5_S5_S5_PKiPKsS7_mEEEvT_DpT0_
        const int knotCount = (dim-2)/3 - 2;
        int ix = 0;
        ix += (x > data[2+3*(ix+1)+2]) * (ix < knotCount); // 1
        ix += (x > data[2+3*(ix+1)+2]) * (ix < knotCount); // 2
        ix += (x > data[2+3*(ix+1)+2]) * (ix < knotCount); // 3
        ix += (x > data[2+3*(ix+1)+2]) * (ix < knotCount); // 4
        ix += (x > data[2+3*(ix+1)+2]) * (ix < knotCount); // 5
        ix += (x > data[2+3*(ix+1)+2]) * (ix < knotCount); // 6
        ix += (x > data[2+3*(ix+1)+2]) * (ix < knotCount); // 7
        ix += (x > data[2+3*(ix+1)+2]) * (ix < knotCount); // 8
        ix += (x > data[2+3*(ix+1)+2]) * (ix < knotCount); // 9
        ix += (x > data[2+3*(ix+1)+2]) * (ix < knotCount); // 10
        ix += (x > data[2+3*(ix+1)+2]) * (ix < knotCount); // 11
        ix += (x > data[2+3*(ix+1)+2]) * (ix < knotCount); // 12
        ix += (x > data[2+3*(ix+1)+2]) * (ix < knotCount); // 13
        ix += (x > data[2+3*(ix+1)+2]) * (ix < knotCount); // 14
        ix += (x > data[2+3*(ix+1)+2]) * (ix < knotCount); // 15
#else /* CALCULATE_GENERAL_SPLINE_BINARY_IF */
        // Check to find a point that is less than x.  This is a "brute force"
        // binary search, and the "if" has been checked and is efficient with
        // CUDA.
// PROFILED aspen 2024/05/26 -- 8:00am (~190 it/sec)
// LTS/Issue510/AtomicOperationsOnHosts w/ GundamInputsOA2021
// commit 5e9415e55b258d3d82d562942f03ac07c9937528
//
// ==1063988== Profiling result:
//             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//                    20.26%  137.925s    262004  526.42us  517.37us  550.90us  _ZN4hemi6KernelIN78_GLOBAL__N__54_tmpxft_00103aad_00000000_7_WeightGeneralSpline_cpp1_ii_220155e217HEMISplinesKernelEJPdPKdS5_S5_S5_PKiPKsS7_mEEEvT_DpT0_
//
        const int knotCount = (dim-2)/3 - 2;
        int ix = 0;
#define CHECK_OFFSET(ioff)  if ((ix+ioff < knotCount) && (x > data[2+3*(ix+ioff)+2])) ix += ioff
        CHECK_OFFSET(16);
        CHECK_OFFSET(8);
        CHECK_OFFSET(4);
        CHECK_OFFSET(2);
        CHECK_OFFSET(1);
#endif

        const double x1 = data[2+3*ix+2];
        const double x2 = data[2+3*(ix+1)+2];
        const double step = x2-x1;

        const double fx = (x - x1)/step;

        const double p1 = data[2+3*ix];
        const double m1 = data[2+3*ix+1]*step;
        const double p2 = data[2+3*(ix+1)];
        const double m2 = data[2+3*(ix+1)+1]*step;

#ifdef DO_NOT_USE_HORNER_FACTORIZATION
        const double fxx = fx*fx;
        const double fxxx = fx*fxx;

        // Cubic spline with the points and slopes.
        // double v = p1*(2.0*fxxx-3.0*fxx+1.0) + m1*(fxxx-2.0*fxx+fx)
        //         + p2*(3.0*fxx-2.0*fxxx) + m2*(fxxx-fxx));

        // A more numerically stable calculation
        const double t = 3.0*fxx-2.0*fxxx;
        double v = p1 - p1*t + m1*(fxxx-2.0*fxx+fx)
                    + p2*t + m2*(fxxx-fxx);
#else
        // Factored via Horner's method.
        double v = ((((2.0*p1 - 2.0*p2 + m2 + m1)*fx
                      + 3.0*p2 - 3.0*p1 - m2 - 2.0*m1)*fx
                     +m1)*fx
                    +p1);
#endif

        if (v < lowerBound) v = lowerBound;
        if (v > upperBound) v = upperBound;

        return v;
    }
}

// An MIT Style License

// Copyright (c) 2022 Clark McGrew

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Local Variables:
// mode:c++
// c-basic-offset:4
// compile-command:"$(git rev-parse --show-toplevel)/cmake/scripts/gundam-build.sh"
// End:
#endif
