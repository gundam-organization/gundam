#ifndef CalculateGraph_h_SEEN
// Calculate a graph specified by the knot position and value.  The knot
// positions must be in increasing order.  This adds a function that can be
// called from CPU (with c++), or a GPU (with CUDA).

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

// Make it easy to override the floating point type.  This would normally be
// done using a typedef, but that doesn't play well with the CUDA compiler.
#ifndef DEVICE_FLOATING_POINT
#define DEVICE_FLOATING_POINT double
#endif

// Place in a private name space so it plays nicely with CUDA
namespace {
    /// Interpolate one point in a graph with non-uniform points.  The graph can
    /// have at most 15 knots defined.
    ///
    /// This takes the parameter value, a minimum and maximum bound, the
    /// buffer of data for this graph, and the number of data elements in the
    /// graph data.  The input data is arrange as
    ///
    /// data[2*n+0] -- The function value for knot n
    /// data[2*n+1] -- The point for knot n
    ///
    /// NOTE: CalculateUniformSpline, CalculateGeneralSpline,
    /// CalculateCompactSpline, and CalculateMonotonicSpline have very similar,
    /// but different calls.  In particular the dim parameter meaning is not
    /// consistent.
    DEVICE_CALLABLE_INLINE
    double CalculateGraph(const double x,
                          const double lowerBound, double upperBound,
                          const DEVICE_FLOATING_POINT* data,
                          const int dim) {

        // Short circuit 1 point graphs.
        if (dim < 4) return data[0];

        // Check to find a point that is less than x.  This is "brute force"
        // binary search for upto 16 elements.  The "if" has been checked and
        // is efficient with CUDA.
        const int knotCount = (dim)/2;
        int ix = 0;
#define CHECK_OFFSET(ioff)  if ((ix+ioff < knotCount) && (x > data[2*(ix+ioff)+1])) ix += ioff
      // __builtin_clz(knotCount) counts the number of leading zeros in knotCount (available in GCC/Clang).
      // sizeof(knotCount) - 1 - __builtin_clz(knotCount) calculates the position of the most significant bit.
      // 1 << (31 - __builtin_clz(knotCount)) generates the largest power of 2 ≤ knotCount
      for( int offset = 1 << ( sizeof(knotCount) - 1 - __builtin_clz(knotCount) ) ; offset > 0 ; offset >>= 1 ){
        CHECK_OFFSET(offset);
      }
#undef CHECK_OFFSET

        // handle positive extrapolation
        // this ensures that x2 exists in the array
        if( ix + 1 >= knotCount ){ ix--; }

        const double p1 = data[2*ix];
        const double x1 = data[2*ix+1];

        const double p2 = data[2*(ix+1)];
        const double x2 = data[2*(ix+1)+1];

        const double step = x2-x1;

        const double fx = (x - x1)/step;

        const double m = p2-p1;

        double v = p1 + fx*m;

        if (v < lowerBound) v = lowerBound;
        if (v > upperBound) v = upperBound;

        return v;
    }
}

#ifdef TEST_CALCULATE_GRAPH
// Compile and test with
//
// cp CalculateGraph.h temp.cpp
// g++ -DTEST_CACULATE_GRAPH temp.cpp
// ./a.out
#include <iostream>
int main(int argc, char** argv) {
    std::cout << "Hello World" << std::endl;
    double data[]{
        0.0, 0.0,
            1.0, 1.0,
            0.0, 2.0,
            2.0, 3.0,
            3.0, 4.0};

    for (double x = -1.0; x<6.0; x += 0.1) {
        double v = CalculateGraph(x,0.0,4.0,data,10);
        std::cout << x << " " << v << std::endl;
    }
    return 0;
}
#endif

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
