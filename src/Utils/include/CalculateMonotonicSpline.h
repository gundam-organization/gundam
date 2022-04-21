#ifndef CALCULATE_UNIFORM_SPLINE_H_SEEN
// Calculate a spline with uniformly space knots.  This adds a function that
// can be called from CPU (with c++), or a GPU (with CUDA).

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
#define DEVICE_CALLABLE_INLINE /* __host__ __device__ */ inline
#endif
#endif

// Allow the floating point type to be overriden.  This would normally be done
// using a typedef, but that doesn't play well with all CUDA compilers.
#ifndef DEVICE_FLOATING_POINT
#define DEVICE_FLOATING_POINT double
#endif

// Place in a private name space so it plays nicely with CUDA
namespace {
    // Interpolate one point using a monotonic spline.  This takes the "index"
    // of the point in the data, the parameter value (that made the index), a
    // minimum and maximum bound, the buffer of data for this spline, and the
    // number of data elements in the spline data.  The input data is arrange
    // as

    // data[0] -- spline lower bound (not used)
    // data[1] -- spline inverse step (not used)
    // data[2+2*n+0] -- The function value for knot n
    DEVICE_CALLABLE_INLINE
    double CalculateMonotonicSpline(const double x,
                                    const double lowerBound, double upperBound,
                                    const DEVICE_FLOATING_POINT* data,
                                    const int dim) {

        // Interpolate between p2 and p3
        // ix-2 ix-1 ix   ix+1 ix+2 ix+3
        // p0   p1   p2---p3   p4   p5
        //   d10  d21  d32  d43  d54
        // m0| |m1| |m2| |m3| |m4| |m5
        //  a0 ||a1 ||a2 ||a3 ||a4 ||a5
        //     b0   b1   b2   b3   b4

        const double low = data[0];
        const double step = 1.0/data[1];

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

        // Get the values of the deltas
        const double d21 = data[2+d21_1] - data[2+d21_0];
        const double d32 = p3-p2;
        const double d43 = data[2+d43_1] - data[2+d43_0];

        // Find the raw slopes at each point
        // double m1 = 0.5*(d10+d21);
        double m2 = 0.5*(d21+d32);
        double m3 = 0.5*(d32+d43);

#define COMPACT_SPLINE_MONOTONIC
#ifdef COMPACT_SPLINE_MONOTONIC
// #warning Using a MONOTONIC spline
        const double d54 = data[2+d54_1] - data[2+d54_0];
        double m4 = 0.5*(d43+d54);

        // Deal with cusp points and flat areas.
        // if (d21*d10 < 0.0) m1 = 0.0;
        if (d32*d21 <= 0.0) m2 = 0.0;
        if (d43*d32 <= 0.0) m3 = 0.0;
        if (d54*d43 <= 0.0) m4 = 0.0;

        // Find the alphas and betas
        // double a0 = (d10>0) ? m0/d21: 0;
        // double b0 = (d10>0) ? m1/d21: 0;
        // double a1 = (d21>0) ? m1/d21: 0;
        const double b1 = (d21>0) ? m2/d21: 0;
        const double a2 = (d32>0) ? m2/d32: 0;
        const double b2 = (d32>0) ? m3/d32: 0;
        const double a3 = (d43>0) ? m3/d43: 0;
        const double b3 = (d43>0) ? m4/d43: 0;
        // double a4 = (d54>0) ? m4/d54: 0;
        // double b4 = (d54>0) ? m5/d54: 0;

        // Find places where can only be piecewise monotonic.
        // if (b0 <= 0) m1 = 0.0;
        if (b1 <= 0) m2 = 0.0;
        if (b2 <= 0) m3 = 0.0;
        // if (b3 <= 0) m4 = 0.0;
        // if (b4 <= 0) m5 = 0.0;
        // if (a0 <= 0) m0 = 0.0;
        // if (a1 <= 0) m1 = 0.0;
        if (a2 <= 0) m2 = 0.0;
        if (a3 <= 0) m3 = 0.0;
        // if (a4 <= 0) m4 = 0.0;

        // Limit the slopes so there isn't overshoot.  It might be
        // possible to handle the zero slope sections here without an
        // extra conditional.  if (a1 > 3 || b1 > 3) m1 = 3.0*d21;
        if (a2 > 3 || b2 > 3) m2 = 3.0*d32;
        if (a3 > 3 || b3 > 3) m3 = 3.0*d43;
        // if (a4 > 3 || b4 > 3) m4 = 3.0*d54;
#endif

        // Cubic spline with the points and slopes.
        double v = (p2*(2.0*fxxx-3.0*fxx+1.0) + m2*(fxxx-2.0*fxx+fx)
                    + p3*(3.0*fxx-2.0*fxxx) + m3*(fxxx-fxx));

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
