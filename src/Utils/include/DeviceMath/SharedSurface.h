#ifndef SHARED_SURFACE_H_SEEN
#define SHARED_SURFACE_H_SEEN

#ifndef GUNDAM_DEVICE_MATH_SKIP_COMMON_INCLUDE
#include "DeviceMath/SharedSplineCommon.h"
#endif

// Uses the Catmull-Rom/Horner interpolation and edge continuation from
// CalculateBicubicSpline.h and CalculateBilinearInterpolation.h.
// Copyright (c) 2024 Clark McGrew
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

namespace GundamDeviceMath {
  // The packed surface is [nx, ny, x[nx], y[ny], knots[nx * ny]].
  // Select a valid segment even at a boundary or during extrapolation. Unlike
  // a fixed 16-knot search, this never reads outside a short coordinate array.
  GUNDAM_DEVICE_MATH_INLINE
  int SurfaceSegment(GUNDAM_DEVICE_MATH_SCALAR value_, GUNDAM_DEVICE_MATH_PTR_CONST axis_, int count_) {
    int low = 0, high = count_ - 1;
    while( low + 1 < high ){
      const int middle = (low + high) / 2;
      if( value_ < axis_[middle] ){ high = middle; }
      else{ low = middle; }
    }
    return low;
  }

  GUNDAM_DEVICE_MATH_INLINE
  GUNDAM_DEVICE_MATH_SCALAR SurfaceCubic(
      GUNDAM_DEVICE_MATH_SCALAR x_,
      GUNDAM_DEVICE_MATH_SCALAR x0_, GUNDAM_DEVICE_MATH_SCALAR p0_,
      GUNDAM_DEVICE_MATH_SCALAR x1_, GUNDAM_DEVICE_MATH_SCALAR p1_,
      GUNDAM_DEVICE_MATH_SCALAR x2_, GUNDAM_DEVICE_MATH_SCALAR p2_,
      GUNDAM_DEVICE_MATH_SCALAR x3_, GUNDAM_DEVICE_MATH_SCALAR p3_) {
    const auto step = x2_ - x1_;
    const auto fraction = (x_ - x1_) / step;
    auto m1 = step * (p2_ - p0_) / (x2_ - x0_);
    auto m2 = step * (p3_ - p1_) / (x3_ - x1_);
    if( fraction < 0 ){ m2 = m1; }
    if( fraction > 1 ){ m1 = m2; }
    return ((((-2 * p2_ + 2 * p1_ + m2 + m1) * fraction
               + 3 * p2_ - 3 * p1_ - m2 - 2 * m1) * fraction + m1) * fraction + p1_);
  }

  GUNDAM_DEVICE_MATH_INLINE
  GUNDAM_DEVICE_MATH_SCALAR SurfaceRowKnot(GUNDAM_DEVICE_MATH_PTR_CONST knots_, int x_, int y_, int ny_) {
    if( y_ < 0 ){ return 2 * knots_[x_ * ny_] - knots_[x_ * ny_ + 1]; }
    if( y_ >= ny_ ){ return 2 * knots_[x_ * ny_ + ny_ - 1] - knots_[x_ * ny_ + ny_ - 2]; }
    return knots_[x_ * ny_ + y_];
  }

  GUNDAM_DEVICE_MATH_INLINE
  GUNDAM_DEVICE_MATH_SCALAR EvaluateSurface(
      GUNDAM_DEVICE_MATH_SCALAR x_, GUNDAM_DEVICE_MATH_SCALAR y_,
      GUNDAM_DEVICE_MATH_PTR_CONST data_, bool cubic_, bool extrapolate_) {
    const int nx = int(data_[0]), ny = int(data_[1]);
    GUNDAM_DEVICE_MATH_PTR_CONST xx = data_ + 2;
    GUNDAM_DEVICE_MATH_PTR_CONST yy = xx + nx;
    GUNDAM_DEVICE_MATH_PTR_CONST knots = yy + ny;
    if( !extrapolate_ ){
      x_ = ClampResponse(x_, xx[0], xx[nx - 1]);
      y_ = ClampResponse(y_, yy[0], yy[ny - 1]);
    }
    const int ix = SurfaceSegment(x_, xx, nx), iy = SurfaceSegment(y_, yy, ny);
    GUNDAM_DEVICE_MATH_SCALAR value;
    if( !cubic_ ){
      const auto first = knots[ix * ny + iy] + (knots[(ix + 1) * ny + iy] - knots[ix * ny + iy])
          * (x_ - xx[ix]) / (xx[ix + 1] - xx[ix]);
      const auto second = knots[ix * ny + iy + 1] + (knots[(ix + 1) * ny + iy + 1] - knots[ix * ny + iy + 1])
          * (x_ - xx[ix]) / (xx[ix + 1] - xx[ix]);
      value = first + (second - first) * (y_ - yy[iy]) / (yy[iy + 1] - yy[iy]);
    }
    else{
      GUNDAM_DEVICE_MATH_SCALAR rows[4];
      const auto x0 = ix > 0 ? xx[ix - 1] : 2 * xx[ix] - xx[ix + 1];
      const auto x3 = ix + 2 < nx ? xx[ix + 2] : 2 * xx[ix + 1] - xx[ix];
      for( int row = 0; row < 4; ++row ){
        const int y = iy + row - 1;
        const auto p1 = SurfaceRowKnot(knots, ix, y, ny);
        const auto p2 = SurfaceRowKnot(knots, ix + 1, y, ny);
        const auto p0 = ix > 0 ? SurfaceRowKnot(knots, ix - 1, y, ny) : 2 * p1 - p2;
        const auto p3 = ix + 2 < nx ? SurfaceRowKnot(knots, ix + 2, y, ny) : 2 * p2 - p1;
        rows[row] = SurfaceCubic(x_, x0, p0, xx[ix], p1, xx[ix + 1], p2, x3, p3);
      }
      const auto y0 = iy > 0 ? yy[iy - 1] : 2 * yy[iy] - yy[iy + 1];
      const auto y3 = iy + 2 < ny ? yy[iy + 2] : 2 * yy[iy + 1] - yy[iy];
      value = SurfaceCubic(y_, y0, rows[0], yy[iy], rows[1], yy[iy + 1], rows[2], y3, rows[3]);
    }
    // Match the native surface's numerical response limits before supervision.
    return ClampResponse(value, GUNDAM_DEVICE_MATH_SCALAR(-1E20), GUNDAM_DEVICE_MATH_SCALAR(1E20));
  }
}
#endif
