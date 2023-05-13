#ifndef CALCULATE_COMPACT_SPLINE_H_SEEN
// Calculate a Catmull-Rom spline with uniformly space knots.  This adds a
// function that can be called from CPU (with c++), or a GPU (with CUDA).
// This is much faster than TSpline3.  This uses the constraint that the slope
// at the first and last point is equal to the average slope between the first
// and last two points [i.e. The slope at the first point, M0, is equal to the
// slope between P0 and P1, so M0 is (P1-P0)/(dist)]. The slope at any point
// is fixed to be the average slope between the preceding and following
// points.

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
// using a typedef, but that doesn't play well with all CUDA compilers.
#ifndef DEVICE_FLOATING_POINT
#define DEVICE_FLOATING_POINT double
#endif

#define GET_VAR_NAME_VALUE(var) ( ((std::stringstream&) (std::stringstream() << #var << " = " << (var)) ).str() )

// Place in a private name space so it plays nicely with CUDA
namespace {
    // Interpolate one point using a compact spline.  This takes the "index"
    // of the point in the data, the parameter value (that made the index), a
    // minimum and maximum bound, the buffer of data for this spline, and the
    // number of data elements in the spline data.  The input data is arrange
    // as

    namespace CompactSplineVars {
      bool DEBUG_PRINOUT{false};
      std::mutex DEBUG_MUTEX{};
    }


    // data[0] -- spline lower bound (not used)
    // data[1] -- spline inverse step (not used)
    // data[2+2*n+0] -- The function value for knot n
    DEVICE_CALLABLE_INLINE
    double CalculateCompactSpline(const double x,
                                  const double lowerBound, double upperBound,
                                  const DEVICE_FLOATING_POINT* data,
                                  const int dim) {
      if( CompactSplineVars::DEBUG_PRINOUT ){
        CompactSplineVars::DEBUG_MUTEX.lock();
      }

        // Interpolate between p2 and p3
        // ix-2 ix-1 ix   ix+1 ix+2 ix+3
        // p0   p1   p2---p3   p4   p5
        //   d10  d21  d32  d43  d54
        // m0| |m1| |m2| |m3| |m4| |m5
        //  a0 ||a1 ||a2 ||a3 ||a4 ||a5
        //     b0   b1   b2   b3   b4

        const double low = data[0];
        const double step = data[1];

        // Get the integer part
        const double xx = (x-low)/step;
        const int ix = xx;

        // Calculate the indices of the two points to calculate d10
        // int d10_0 = ix-2;             // p0
        // if (d10_0<0) d10_0 = 0;
        // int d10_1 = d10_0+1;          // p1
        // Calculate the indices of the two points to calculate d21
        int d21_0 = ix-1;             // p1
        if (d21_0 < 0)     d21_0 = 0;
        if (d21_0 > dim-2) d21_0 = dim-2;
        int d21_1 = d21_0+1;          // p2
        // Calculate the indices of the two points to calculate d21
        int d32_0 = ix;               // p2
        if (d32_0 < 0)     d32_0 = 0;
        if (d32_0 > dim-2) d32_0 = dim-2;
        int d32_1 = d32_0+1;          // p3
        // Calculate the indices of the two points to calculate d43;
        int d43_0 = ix+1;             // p3
        if (d43_0 < 0)     d43_0 = 0;
        if (d43_0 > dim-2) d43_0 = dim-2;
        int d43_1 = d43_0+1;          // p4
        // Calculate the indices of the two points to calculate d43;
        int d54_0 = ix+2;             // p4
        if (d54_0 < 0)     d54_0 = 0;
        if (d54_0 > dim-2) d54_0 = dim-2;
        int d54_1 = d54_0+1;          // p5

        // Find the points to use.
        const double p2 = data[2+d32_0];
        const double p3 = data[2+d32_1];

        // Get the remainder for the "index".  If the input index is less than
        // zero or greater than kPointSize-1, this is the distance from the
        // boundary.
        const double fx = xx-d32_0;
        const double fxx = fx*fx;
        const double fxxx = fx*fxx;
        LogTraceIf(CompactSplineVars::DEBUG_PRINOUT) << GET_VAR_NAME_VALUE(fx) << std::endl;
        LogTraceIf(CompactSplineVars::DEBUG_PRINOUT) << GET_VAR_NAME_VALUE(fxx) << std::endl;
        LogTraceIf(CompactSplineVars::DEBUG_PRINOUT) << GET_VAR_NAME_VALUE(fxxx) << std::endl;

        // Get the values of the deltas
        const double d21 = data[2+d21_1] - data[2+d21_0];
        const double d32 = p3-p2;
        const double d43 = data[2+d43_1] - data[2+d43_0];
        LogTraceIf(CompactSplineVars::DEBUG_PRINOUT) << GET_VAR_NAME_VALUE(d21) << std::endl;
        LogTraceIf(CompactSplineVars::DEBUG_PRINOUT) << GET_VAR_NAME_VALUE(d32) << std::endl;
        LogTraceIf(CompactSplineVars::DEBUG_PRINOUT) << GET_VAR_NAME_VALUE(d43) << std::endl;
        LogTraceIf(CompactSplineVars::DEBUG_PRINOUT) << GET_VAR_NAME_VALUE(data[2+d43_1]) << std::endl;
        LogTraceIf(CompactSplineVars::DEBUG_PRINOUT) << GET_VAR_NAME_VALUE(data[2+d43_0]) << std::endl;

        // Find the raw slopes at each point
        // double m1 = 0.5*(d10+d21);
        double m2 = 0.5*(d21+d32);
        double m3 = 0.5*(d32+d43);
        LogTraceIf(CompactSplineVars::DEBUG_PRINOUT) << GET_VAR_NAME_VALUE(m2) << std::endl;
        LogTraceIf(CompactSplineVars::DEBUG_PRINOUT) << GET_VAR_NAME_VALUE(m3) << std::endl;

        // Cubic spline with the points and slopes.
        // double v = p2*(2.0*fxxx-3.0*fxx+1.0) + m2*(fxxx-2.0*fxx+fx)
        //         + p3*(3.0*fxx-2.0*fxxx) + m3*(fxxx-fxx));

        // A more numerically stable calculation
        const double t = 3.0*fxx-2.0*fxxx;
        double v = p2 - p2*t + m2*(fxxx-2.0*fxx+fx)
                    + p3*t + m3*(fxxx-fxx);

        LogTraceIf(CompactSplineVars::DEBUG_PRINOUT) << GET_VAR_NAME_VALUE(v) << std::endl;

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
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
#endif
