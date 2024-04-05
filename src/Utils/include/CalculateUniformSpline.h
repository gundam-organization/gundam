#ifndef CALCULATE_UNIFORM_SPLINE_H_SEEN
// Calculate a spline with uniformly space knots.  This adds a function that
// can be called from CPU (with c++), or a GPU (with CUDA).  This is much
// faster than TSpline3.

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
    // Interpolate one point using a spline with uniformly spaced knots.  This
    // is much faster than TSpline3 (about fifty times faster, but careful,
    // controlled benchmarking was not done).  This takes the "index" of the
    // point in the data, the parameter value (that made the index), a minimum
    // and maximum output value, the buffer of data for this spline, and the
    // number of data elements in the spline data.  The input data is arrange
    // as
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
    DEVICE_CALLABLE_INLINE
    double CalculateUniformSpline(const double x,
                                  const double lowerBound, double upperBound,
                                  const DEVICE_FLOATING_POINT* data,
                                  const int dim) {

        // Get the integer part
        const double step = data[1];
        const double xx = (x-data[0])/step;
        int ix = xx;
        if (ix<0) ix=0;
        if (2*ix+7>dim) ix = (dim-2)/2 - 2 ;

        const double fx = xx-ix;

        const double p1 = data[2+2*ix];
        const double m1 = data[2+2*ix+1]*step;
        const double p2 = data[2+2*ix+2];
        const double m2 = data[2+2*ix+3]*step;

        // Cubic spline with the points and slopes.
        // double v = p1*(2.0*fxxx-3.0*fxx+1.0) + m1*(fxxx-2.0*fxx+fx)
        //         + p2*(3.0*fxx-2.0*fxxx) + m2*(fxxx-fxx));

        // Factored via Horner's method.
        double v = ((((2.0*p1 - 2.0*p2 + m2 + m1)*fx
                      + 3.0*p2 - 3.0*p1 - m2 - 2.0*m1)*fx
                     +m1)*fx
                    +p1);

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
