#ifndef CALCULATE_BILINEAR_INTERPOLATION_H_SEEN
#include <stdio.h>

// Calculate a bilinear (2D) interpolation specified by the position, and
// value at each knot.  This adds a local function that can be called from CPU
// (with c++), or a GPU (with CUDA).

// Wrap the CUDA compiler attributes into a definition.  When this is compiled
// with a CUDA compiler __CUDACC__ will be defined.  In that case, the code
// will be compiled with cuda attributes for both the host (i.e. __host__) and
// gpu (i.e. __device__).  If it's compiled with a normal C compiler, this is
// compiled as an inline function
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
// using a typedef, but that doesn't play well with the CUDA compiler.
#ifndef DEVICE_FLOATING_POINT
#define DEVICE_FLOATING_POINT double
#endif

// Place in a private name space so it plays nicely with CUDA
namespace {

    // Find the index of the first point in the array a[na] that is less than
    // or equal to v ("a" is a pointer to the data, and "na" is the number of
    // entries in the data). This works up to 16 entries. The values in "a"
    // must in strictly increasing order.
    DEVICE_CALLABLE_INLINE
    int BilinearIndex(const DEVICE_FLOATING_POINT v,
                      const DEVICE_FLOATING_POINT* a,
                      const int na) {
        // Do a binary search:
        //
        // * This is not obfuscated code, but a "standard" way to avoid
        //   branches for SIMD code (i.e. on a GPU).
        //
        // * Use the C++ definition that a boolean cast as an integer is equal
        //   to zero or one.
        //
        // * Use "not <" instead of "<=" to make sure compiler doesn't use a
        //   branch.
        //
        // * There _are_ slightly faster ways of doing this, but they are even
        //   trickier.
        // int i = (not (v < a[i+8])) * ((i+8)<na) * 8;
        int i = (not (v < a[8])) * ((8)<na) * 8;
        i += (not (v < a[i+4])) * ((i+4)<na) * 4;
        i += (not (v < a[i+2])) * ((i+2)<na) * 2;
        i += (not (v < a[i+1])) * ((i+1)<na) * 1;
        return i;
    }

    // Do the "interpolation" in one dimension
    DEVICE_CALLABLE_INLINE
    DEVICE_FLOATING_POINT
    BilinearCINT(const DEVICE_FLOATING_POINT x,
                 const DEVICE_FLOATING_POINT x0,
                 const DEVICE_FLOATING_POINT p0,
                 const DEVICE_FLOATING_POINT x1,
                 const DEVICE_FLOATING_POINT p1) {
        return p0 + (p1-p0)*(x-x0)/(x1-x0);
    }

    // Interpolate one point with non-uniform knots on each axis.  Each axis
    // can have have at most 16 knots defined, and there must be at least 4
    // knots (i.e. a minimum of one square facet)..  The knots are specified
    // on a grid [0 .. NX-1] x [0 .. NY-1], and the absicssa coordinates of
    // the knots are specified in two arrays [0 .. NX-1] and [0 .. Ny-1]
    //
    // The knots and coordinates of the knots are specified in the data block
    //
    // knots[Nx*Ny]   -- The grid of knots
    // xx[NNx]        -- The x coordinates
    // yy[NNy]        -- The y coordinates
    //
    // The values of Nx (Ny), and NNx (NNy) should normally be equal, but are
    // provided separately so that the same calling sequence can be used for
    // uniform bilinear interpolations.  If Nx (Ny), and NNx (NNy) are not
    // equal, then the value of NNx (NNy) should be 2 and the x (y)
    // coordinates will be the bounds of the grid (the step will be
    // (xx[1]-xx[0])/(Nx-1)).
    //
    // Example: For a uniform 3 x 3 grid from -1 to 1 with a step size of 1,
    // the knots are at:
    //
    // (-1,-1) (-1, 0) (-1, 1)
    // ( 0,-1) ( 0, 0) ( 0, 1)
    // ( 1,-1) ( 1, 0) ( 1, 1)
    //
    // The value of nx is 3, and the value of ny is 3.
    //
    // The values in xx will be [-1, 0, 1] and nnx will be 3
    // The values in yy will be [-1, 0, 1] and nny will be 3
    //
    DEVICE_CALLABLE_INLINE
    DEVICE_FLOATING_POINT
    CalculateBilinearInterpolation(const DEVICE_FLOATING_POINT x,
                                   const DEVICE_FLOATING_POINT y,
                                   const DEVICE_FLOATING_POINT lowerBound,
                                   const DEVICE_FLOATING_POINT upperBound,
                                   const DEVICE_FLOATING_POINT* knots,
                                   const int nx, const int ny,
                                   const DEVICE_FLOATING_POINT* xx,
                                   const int nnx,
                                   const DEVICE_FLOATING_POINT* yy,
                                   const int nny) {

        // Find the indices for the lower corner of the facet used for
        // interpolation.  The point may not lie in the facet and the result
        // will be extrapolated.  The corners of the facet [(ix,iy),
        // (ix+1,iy), (ix,iy+1), and (ix+1,iy+1)] always exist in the grid of
        // knots.
        const int ix = BilinearIndex(x, xx, nnx-1);
        const int iy = BilinearIndex(y, yy, nny-1);

#define IXY(ixx,iyy) ((ixx)*ny + (iyy))
        DEVICE_FLOATING_POINT a0, b0, a1, b1;
        a0 = xx[ix];
        b0 = knots[IXY(ix,iy)];
        a1 = xx[ix+1];
        b1 = knots[IXY(ix+1,iy)];

        const DEVICE_FLOATING_POINT u0 = yy[iy];
        const DEVICE_FLOATING_POINT v0 = BilinearCINT(x,a0,b0,a1,b1);

        a0 = xx[ix];
        b0 = knots[IXY(ix,iy+1)];
        a1 = xx[ix+1];
        b1 = knots[IXY(ix+1,iy+1)];

        const DEVICE_FLOATING_POINT u1 = yy[iy+1];
        const DEVICE_FLOATING_POINT v1 = BilinearCINT(x,a0,b0,a1,b1);

        DEVICE_FLOATING_POINT val = BilinearCINT(y,u0,v0,u1,v1);

        // Apply the clamp.  This could be done using the CUDA min/max
        // primitives, but this is not a critical section of the code, and the
        // min/max api is drawn from "C", not std::max/std::min, so I think
        // this is safer for most users to read.
        if (val<lowerBound) val = lowerBound;
        if (val>upperBound) val = upperBound;

        return val;
    }
}

// An MIT Style License

// Copyright (c) 2024 Clark McGrew

// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

// Local Variables:
// mode:c++
// c-basic-offset:4
// compile-command:"$(git rev-parse --show-toplevel)/cmake/scripts/gundam-build.sh"
// End:
#endif
