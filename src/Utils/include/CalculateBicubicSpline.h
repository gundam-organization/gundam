#ifndef CALCULATE_BICUBIC_SPLINE_H_SEEN
#include <stdio.h>

// Calculate a Bicubic (2D) spline interpolation specified by the position,
// and value at each knot.  This is implemented using the Catmull-Rom spline.
// This adds a local function that can be called from CPU (with c++), or a GPU
// (with CUDA).

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
    int BicubicIndex(const DEVICE_FLOATING_POINT v,
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
    BicubicCINT(const DEVICE_FLOATING_POINT x,
                const DEVICE_FLOATING_POINT x0,
                const DEVICE_FLOATING_POINT p0,
                const DEVICE_FLOATING_POINT x1,
                const DEVICE_FLOATING_POINT p1,
                const DEVICE_FLOATING_POINT x2,
                const DEVICE_FLOATING_POINT p2,
                const DEVICE_FLOATING_POINT x3,
                const DEVICE_FLOATING_POINT p3) {

        const double step = x2-x1;
        const double fx = (x - x1)/step;
        double m1 = step*(p2-p0)/(x2-x0);
        double m2 = step*(p3-p1)/(x3-x1);

        // Make linear outside of region
        if (fx < 0.0) m2 = m1;
        if (fx > 1.0) m1 = m2;

#ifdef BICUBIC_USE_RAW_HERMITIAN
        // Cubic spline with the points and slopes.
        const double fxx = fx*fx;
        const double fxxx = fx*fxx;
        double v1 = p1*(2.0*fxxx-3.0*fxx+1.0) + m1*(fxxx-2.0*fxx+fx)
            + p2*(3.0*fxx-2.0*fxxx) + m2*(fxxx-fxx);
#endif

#ifdef BICUBIC_USE_STABLIZED
        // Cubic spline with the points and slopes.
        const double fxx = fx*fx;
        const double fxxx = fx*fxx;
        const double t = 3.0*fxx-2.0*fxxx;
        double v = p1 - p1*t + m1*(fxxx-2.0*fxx+fx)
            + p2*t + m2*(fxxx-fxx);
#endif

#define BICUBIC_USE_HORNER_FACTORIZATION
#ifdef BICUBIC_USE_HORNER_FACTORIZATION
        double v = ((((-2.0*p2 + 2.0*p1 + m2 + m1)*fx
                      +3*p2-3*p1-m2-2*m1)*fx
                     + m1)*fx
                    + p1);
#endif

        return v;

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
    // uniform bicubic splines.  If Nx (Ny), and NNx (NNy) are not equal, then
    // the value of NNx (NNy) should be 2 and the x (y) coordinates will be
    // the bounds of the grid (the step will be (xx[1]-xx[0])/(Nx-1)).
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
    CalculateBicubicSpline(const DEVICE_FLOATING_POINT x,
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
        const int ix = BicubicIndex(x, xx, nnx-1);
        const int iy = BicubicIndex(y, yy, nny-1);

        // First interpolate along the "x" axis at each "y" value.  This will
        // find 4 interpolated values (v0, v1, v2, v3) at 3 different y values
        // (u0, u1, u2, u3).
#define IXY(ixx,iyy) ((ixx)*ny + (iyy))

        DEVICE_FLOATING_POINT a0, b0, a1, b1, a2, b2, a3, b3;

        // Interpolate the first band (at a fixed y value) in yy[iy-1].
        // Handle the corner cases.
        a1 = xx[ix];
        if (iy>0) b1 = knots[IXY(ix,iy-1)];   // Y ok
        else b1 = 2*knots[IXY(ix,iy)] - knots[IXY(ix,iy+1)]; // Y low
        a2 = xx[ix+1];
        if (iy>0) b2 = knots[IXY(ix+1,iy-1)];   // Y ok
        else b2 = 2*knots[IXY(ix+1,iy)] - knots[IXY(ix+1,iy+1)]; // Y low
        if (ix>0) {   // X ok
            a0 = xx[ix-1];
            if (iy>0) b0 = knots[IXY(ix-1,iy-1)];  // Y ok
            else b0 = 2*knots[IXY(ix-1,iy)] - knots[IXY(ix-1,iy+1)]; // Y low
        }
        else {       // X low
            a0 = 2*xx[ix] - xx[ix+1];
            if (iy>0) {
                b0 = 2*knots[IXY(ix,iy-1)] - knots[IXY(ix+1,iy-1)]; // Y ok
            }
            else {
                b0 = 2*b1 - b2; // Y low
            }
        }
        if (ix < nnx-2) {       // X ok
            a3 = xx[ix+2];
            if (iy>0) b3 = knots[IXY(ix+2,iy-1)];  // Y ok
            else b3 = 2*knots[IXY(ix+2,iy)] - knots[IXY(ix+2,iy+1)]; // Y low
        }
        else {                  // X high
            a3 = 2*xx[ix+1] - xx[ix];
            if (iy>0) b3 = 2*knots[IXY(ix+1,iy-1)] - knots[IXY(ix,iy-1)]; // Y ok
            else b3 = 2*b2 - b1; // Y ow
        }

        const DEVICE_FLOATING_POINT u0 = (iy > 0) ? yy[iy-1]: 2*yy[iy]-yy[iy+1];
        const DEVICE_FLOATING_POINT v0 = BicubicCINT(x,a0,b0,a1,b1,a2,b2,a3,b3);

        // Interpolate the second band (at a fixed y value) in yy[iy].
        // Handle the corner cases.
        a1 = xx[ix];
        b1 = knots[IXY(ix,iy)];
        a2 = xx[ix+1];
        b2 = knots[IXY(ix+1,iy)];
        if (ix > 0) {    // X ok
            a0 = xx[ix-1];
            b0 = knots[IXY(ix-1,iy)];
        }
        else {  // X low
            a0 = 2*xx[ix] - xx[ix+1];
            b0 = 2*knots[IXY(ix,iy)] - knots[IXY(ix+1,iy)];
        }
        if (ix < nnx - 2) {  // X ok
            a3 = xx[ix+2];
            b3 = knots[IXY(ix+2,iy)];
        }
        else { // X high
            a3 = 2*xx[ix+1] - xx[ix];
            b3 = 2*knots[IXY(ix+1,iy)] - knots[IXY(ix,iy)];
        }

        const DEVICE_FLOATING_POINT u1 = yy[iy];
        const DEVICE_FLOATING_POINT v1 = BicubicCINT(x,a0,b0,a1,b1,a2,b2,a3,b3);

        // Interpolate the third band (at a fixed y value) in yy[iy+1].
        // Handle the corner cases.
        a1 = xx[ix];
        b1 = knots[IXY(ix,iy+1)];
        a2 = xx[ix+1];
        b2 = knots[IXY(ix+1,iy+1)];
        if (ix > 0) { // X ok
            a0 = xx[ix-1];
            b0 = knots[IXY(ix-1,iy+1)];
        }
        else { // X low
            a0 = 2*xx[ix] - xx[ix+1];
            b0 = 2*knots[IXY(ix,iy+1)] - knots[IXY(ix+1,iy+1)];
        }
        if (ix < nnx-2) {  // X ok
            a3 = xx[ix+2];
            b3 = knots[IXY(ix+2,iy+1)];
        }
        else { // X high
            a3 = 2*xx[ix+1] - xx[ix];
            b3 = 2*knots[IXY(ix+1,iy+1)] - knots[IXY(ix,iy+1)];
        }

        const DEVICE_FLOATING_POINT u2 = yy[iy+1];
        const DEVICE_FLOATING_POINT v2 = BicubicCINT(x,a0,b0,a1,b1,a2,b2,a3,b3);

        // Interpolate the fourth band (at a fixed y value) in yy[iy+2].
        // Handle the corner cases.
        a1 = xx[ix];
        if (iy < nny-2) b1 = knots[IXY(ix,iy+2)]; // Y ok
        else b1 = 2*knots[IXY(ix,iy+1)] - knots[IXY(ix,iy)];  // Y high
        a2 = xx[ix+1];
        if (iy < nny-2) b2 = knots[IXY(ix+1,iy+2)]; // Y ok
        else b2 = 2*knots[IXY(ix+1,iy+1)] - knots[IXY(ix+1,iy)];  // Y high
        if (ix > 0) { // X ok
            a0 = xx[ix-1];
            if (iy < nny-2)  b0 = knots[IXY(ix-1,iy+2)];  // Y ok
            else b0 = 2*knots[IXY(ix-1,iy+1)] - knots[IXY(ix-1,iy)]; // Y high
        }
        else { // X low
            a0 = 2*xx[ix] - xx[ix+1];
            if (iy < nny-2) b0 = 2*knots[IXY(ix,iy+2)] - knots[IXY(ix+1,iy+2)]; // Y ok
            else b0 = 2*b1 - b2; // Y high
        }
        if (ix < nnx-2) { // X ok
            a3 = xx[ix+2];
            if (iy < nny-2) b3 = knots[IXY(ix+2,iy+2)]; // Y ok
            else b3 = 2*knots[IXY(ix+2,iy+1)] - knots[IXY(ix+2,iy)];
        }
        else { // X high
            a3 = 2*xx[ix+1] - xx[ix];
            if (iy < nny-2) b3 = 2*knots[IXY(ix+1,iy+2)] - knots[IXY(ix,iy+2)];
            else b3 = 2*b2 - b1;
        }

        const DEVICE_FLOATING_POINT u3 = (iy < nny-2) ? yy[iy+2] : 2*yy[iy+1] - yy[iy];
        const DEVICE_FLOATING_POINT v3 = BicubicCINT(x,a0,b0,a1,b1,a2,b2,a3,b3);

        DEVICE_FLOATING_POINT val = BicubicCINT(y,u0,v0,u1,v1,u2,v2,u3,v3);

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
