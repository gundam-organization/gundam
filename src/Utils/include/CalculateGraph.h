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

#include "DeviceMath/SharedGraph.h"

namespace {
  DEVICE_CALLABLE_INLINE
  double CalculateGraph(const double x,
                        const double lowerBound, double upperBound,
                        const DEVICE_FLOATING_POINT* data,
                        const int dim) {
    return GundamDeviceMath::EvaluateGraph(x, lowerBound, upperBound, data, dim);
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
