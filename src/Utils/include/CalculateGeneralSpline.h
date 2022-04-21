#ifndef CALCULATE_GENERAL_SPLINE_H_SEEN
// Calculate a spline knots.  This adds a function that can be called from CPU
// (with c++), or a GPU (with CUDA).  With G++ running with O1, this is about
// forty times faster than TSpline3.

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
    // (that made the index), a minimum and maximum bound, the buffer of data
    // for this spline, and the number of data elements in the spline data.
    // The input data is arrange as
    //
    // data[0] -- spline lower bound (not used)
    // data[1] -- spline inverse step (not used)
    // data[2+3*n+0] -- The function value for knot n
    // data[2+3*n+1] -- The function slope for knot n
    // data[2+3*n+2] -- The point for knot n
    DEVICE_CALLABLE_INLINE
    double CalculateGeneralSpline(const double x,
                                  const double lowerBound, double upperBound,
                                  const DEVICE_FLOATING_POINT* data,
                                  const int dim) {

        // Check to find a point that is less than x.
        const int knotCount = (dim-2)/3;
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

        const double x1 = data[2+3*ix+2];
        const double x2 = data[2+3*(ix+1)+2];
        const double step = x2-x1;

        const double fx = (x - x1)/step;
        const double fxx = fx*fx;
        const double fxxx = fx*fxx;

        const double p1 = data[2+3*ix];
        const double m1 = data[2+3*ix+1]*step;
        const double p2 = data[2+3*(ix+1)];
        const double m2 = data[2+3*(ix+1)+1]*step;

        // Cubic spline with the points and slopes.
        double v = (p1*(2.0*fxxx-3.0*fxx+1.0) + m1*(fxxx-2.0*fxx+fx)
                    + p2*(3.0*fxx-2.0*fxxx) + m2*(fxxx-fxx));

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
