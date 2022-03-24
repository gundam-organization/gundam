#include "CacheWeights.h"
#include "WeightBase.h"
#include "WeightCompactSpline.h"

#include <algorithm>
#include <iostream>
#include <exception>
#include <limits>
#include <cmath>

#include <hemi/hemi_error.h>
#include <hemi/launch.h>
#include <hemi/grid_stride_range.h>

#ifndef LOGGER
#define LOGGER std::cout
#endif

// The constructor
Cache::Weight::CompactSpline::CompactSpline(
    Cache::Weights::Results& weights,
    Cache::Parameters::Values& parameters,
    Cache::Parameters::Clamps& lowerClamps,
    Cache::Parameters::Clamps& upperClamps,
    std::size_t splines, std::size_t knots)
    : Cache::Weight::Base("compactSpline",weights,parameters),
      fLowerClamp(lowerClamps), fUpperClamp(upperClamps),
      fSplinesReserved(splines), fSplinesUsed(0),
      fSplineKnotsReserved(knots), fSplineKnotsUsed(0) {

    LOGGER << "Reserved " << GetName() << " Splines: "
           << GetSplinesReserved() << std::endl;
    if (GetSplinesReserved() < 1) return;

    fTotalBytes += GetSplinesReserved()*sizeof(int);      // fSplineResult
    fTotalBytes += GetSplinesReserved()*sizeof(short);    // fSplineParameter
    fTotalBytes += (1+GetSplinesReserved())*sizeof(int);  // fSplineIndex

    fSplineKnotsReserved = 2*fSplinesReserved + fSplineKnotsReserved;

#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning Using SLOW VALIDATION in Cache::Weight::CompactSpline::CompactSpline
        // Add validation code for the spline calculation.  This can be rather
        // slow, so do not use if it is not required.
    fTotalBytes += GetSplinesReserved()*sizeof(double);
#endif

    LOGGER << "Reserved Spline Knots: " << GetSplineKnotsReserved()<< std::endl;
    fTotalBytes += GetSplineKnotsReserved()*sizeof(float);  // fSpineKnots

    LOGGER << "Approximate Memory Size: " << fTotalBytes/1E+9
              << " GB" << std::endl;

    try {
        // Get the CPU/GPU memory for the spline index tables.  These are
        // copied once during initialization so do not pin the CPU memory into
        // the page set.
        fSplineResult.reset(new hemi::Array<int>(GetSplinesReserved(),false));
        fSplineParameter.reset(
            new hemi::Array<short>(GetSplinesReserved(),false));
        fSplineIndex.reset(new hemi::Array<int>(1+GetSplinesReserved(),false));

#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning Using SLOW VALIDATION in Cache::Weight::CompactSpline::CompactSpline
        // Add validation code for the spline calculation.  This can be rather
        // slow, so do not use if it is not required.
        fSplineValue.reset(new hemi::Array<double>(GetSplinesReserved(),true));
#endif

        // Get the CPU/GPU memory for the spline knots.  This is copied once
        // during initialization so do not pin the CPU memory into the page
        // set.
        fSplineKnots.reset(
            new hemi::Array<float>(GetSplineKnotsReserved(),false));
    }
    catch (std::bad_alloc&) {
        LOGGER << "Failed to allocate memory, so stopping" << std::endl;
        throw std::runtime_error("Not enough memory available");
    }

    // Initialize the caches.  Don't try to zero everything since the
    // caches can be huge.
    fSplineIndex->hostPtr()[0] = 0;
}

// The destructor
Cache::Weight::CompactSpline::~CompactSpline() {}

int Cache::Weight::CompactSpline::FindPoints(const TSpline3* s) {
    return s->GetNp();
}

void Cache::Weight::CompactSpline::AddSpline(int resultIndex,
                                             int parIndex,
                                             SplineDial* sDial) {
    const TSpline3* s = sDial->getSplinePtr();
    int NP = Cache::Weight::CompactSpline::FindPoints(s);
    double xMin = s->GetXmin();
    double xMax = s->GetXmax();
    int sIndex = ReserveSpline(resultIndex,parIndex,xMin,xMax,NP);
#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning Using SLOW VALIDATION in Cache::Weight::CompactSpline::AddSpline
    sDial->setGPUCachePointer(GetCachePointer(sIndex));
#endif
    for (int i=0; i<NP; ++i) {
        double x = xMin + i*(xMax-xMin)/(NP-1);
        double y = s->Eval(x);
        SetSplineKnot(sIndex,i,y);
    }
}

// Reserve space in the internal structures for spline with uniform knots.
// The knot values will still need to be set using set spline knot.
int Cache::Weight::CompactSpline::ReserveSpline(
    int resIndex, int parIndex, double low, double high, int points) {
    if (resIndex < 0) {
        LOGGER << "Invalid result index"
               << std::endl;
        throw std::runtime_error("Negative result index");
    }
    if (fWeights.size() <= resIndex) {
        LOGGER << "Invalid result index"
               << std::endl;
        throw std::runtime_error("Result index out of bounds");
    }
    if (parIndex < 0) {
        LOGGER << "Invalid parameter index"
               << std::endl;
        throw std::runtime_error("Negative parameter index");
    }
    if (fParameters.size() <= parIndex) {
        LOGGER << "Invalid parameter index"
               << std::endl;
        throw std::runtime_error("Parameter index out of bounds");
    }
    if (high <= low) {
        LOGGER << "Invalid spline bounds"
               << std::endl;
        throw std::runtime_error("Invalid spline bounds");
    }
    if (points < 3) {
        LOGGER << "Insufficient points in spline"
               << std::endl;
        throw std::runtime_error("Invalid number of spline points");
    }
    int newIndex = fSplinesUsed++;
    if (fSplinesUsed > fSplinesReserved) {
        LOGGER << "Not enough space reserved for splines"
                  << std::endl;
        throw std::runtime_error("Not enough space reserved for splines");
    }
    fSplineResult->hostPtr()[newIndex] = resIndex;
    fSplineParameter->hostPtr()[newIndex] = parIndex;
    if (fSplineIndex->hostPtr()[newIndex] != fSplineKnotsUsed) {
        LOGGER << "Last spline knot index should be at old end of splines"
                  << std::endl;
        throw std::runtime_error("Problem with control indices");
    }
    int knotIndex = fSplineKnotsUsed;
    fSplineKnotsUsed += 2; // Space for the upper and lower bound
    fSplineKnotsUsed += points; // Space for the knots.
    if (fSplineKnotsUsed > fSplineKnotsReserved) {
        LOGGER << "Not enough space reserved for spline knots"
               << std::endl;
        throw std::runtime_error("Not enough space reserved for spline knots");
    }
    fSplineIndex->hostPtr()[newIndex+1] = fSplineKnotsUsed;
    // Save values needed to calculate the spline offset index.  If the input
    // value is x, the index is
    // v = (x-CD[dataIndex])*CD[dataIndex+1].
    // i = v;
    // v = v - i;
    double invStep =  1.0*(points-1.0)/(high-low);
    fSplineKnots->hostPtr()[knotIndex] = low;
    fSplineKnots->hostPtr()[knotIndex+1] = invStep;

    return newIndex;
}

void Cache::Weight::CompactSpline::SetSplineKnot(
    int sIndex, int kIndex, double value) {
    if (sIndex < 0) {
        LOGGER << "Requested spline index is negative"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    if (GetSplinesUsed() <= sIndex) {
        LOGGER << "Requested spline index is to large"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    if (kIndex < 0) {
        LOGGER << "Requested control point index is negative"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    int knotIndex = fSplineIndex->hostPtr()[sIndex] + 2 + kIndex;
    if (fSplineIndex->hostPtr()[sIndex+1] <= knotIndex) {
        LOGGER << "Requested control point index is two large"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    fSplineKnots->hostPtr()[knotIndex] = value;
}

int Cache::Weight::CompactSpline::GetSplineParameterIndex(int sIndex) {
    if (sIndex < 0) {
        throw std::runtime_error("Spline index invalid");
    }
    if (GetSplinesUsed() <= sIndex) {
        throw std::runtime_error("Spline index invalid");
    }
    return fSplineParameter->hostPtr()[sIndex];
}

double Cache::Weight::CompactSpline::GetSplineParameter(int sIndex) {
    int i = GetSplineParameterIndex(sIndex);
    if (i<0) {
        throw std::runtime_error("Spine parameter index out of bounds");
    }
    if (fParameters.size() <= i) {
        throw std::runtime_error("Spine parameter index out of bounds");
    }
    return fParameters.hostPtr()[i];
}

int Cache::Weight::CompactSpline::GetSplineKnotCount(int sIndex) {
    if (sIndex < 0) {
        throw std::runtime_error("Spline index invalid");
    }
    if (GetSplinesUsed() <= sIndex) {
        throw std::runtime_error("Spline index invalid");
    }
    return fSplineIndex->hostPtr()[sIndex+1]-fSplineIndex->hostPtr()[sIndex]-2;
}

double Cache::Weight::CompactSpline::GetSplineLowerBound(int sIndex) {
    if (sIndex < 0) {
        throw std::runtime_error("Spline index invalid");
    }
    if (GetSplinesUsed() <= sIndex) {
        throw std::runtime_error("Spline index invalid");
    }
    int knotsIndex = fSplineIndex->hostPtr()[sIndex];
    return fSplineKnots->hostPtr()[knotsIndex];
}

double Cache::Weight::CompactSpline::GetSplineUpperBound(int sIndex) {
    if (sIndex < 0) {
        throw std::runtime_error("Spline index invalid");
    }
    if (GetSplinesUsed() <= sIndex) {
        throw std::runtime_error("Spline index invalid");
    }
    int knotCount = GetSplineKnotCount(sIndex);
    double lower = GetSplineLowerBound(sIndex);
    int knotsIndex = fSplineIndex->hostPtr()[sIndex];
    double step = fSplineKnots->hostPtr()[knotsIndex+1];
    return lower + (knotCount-1)/step;
}

double Cache::Weight::CompactSpline::GetSplineLowerClamp(int sIndex) {
    int i = GetSplineParameterIndex(sIndex);
    if (i<0) {
        throw std::runtime_error("Spine lower clamp index out of bounds");
    }
    if (fLowerClamp.size() <= i) {
        throw std::runtime_error("Spine lower clamp index out of bounds");
    }
    return fLowerClamp.hostPtr()[i];
}

double Cache::Weight::CompactSpline::GetSplineUpperClamp(int sIndex) {
    int i = GetSplineParameterIndex(sIndex);
    if (i<0) {
        throw std::runtime_error("Spine upper clamp index out of bounds");
    }
    if (fUpperClamp.size() <= i) {
        throw std::runtime_error("Spine upper clamp index out of bounds");
    }
    return fUpperClamp.hostPtr()[i];
}

double Cache::Weight::CompactSpline::GetSplineKnot(int sIndex, int knot) {
    if (sIndex < 0) {
        throw std::runtime_error("Spline index invalid");
    }
    if (GetSplinesUsed() <= sIndex) {
        throw std::runtime_error("Spline index invalid");
    }
    int knotsIndex = fSplineIndex->hostPtr()[sIndex];
    int count = GetSplineKnotCount(sIndex);
    if (knot < 0) {
        throw std::runtime_error("Knot index invalid");
    }
    if (count <= knot) {
        throw std::runtime_error("Knot index invalid");
    }
    return fSplineKnots->hostPtr()[knotsIndex+2+knot];
}

////////////////////////////////////////////////////////////////////
// This section is for the validation methods.  They should mostly be
// NOOPs and should mostly not be called.

#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning Using SLOW VALIDATION in Cache::Weight::CompactSpline::GetSplineValue
// Get the intermediate spline result that is used to calculate an event
// weight.  This can trigger a copy from the GPU to CPU, and must only be
// enabled during validation.  Using this validation code also significantly
// increases the amount of GPU memory required.  In a short sentence, "Do not
// use this method."
double* Cache::Weight::CompactSpline::GetCachePointer(int sIndex) {
    if (sIndex < 0) {
        throw std::runtime_error("GetSplineValue: Spline index invalid");
    }
    if (GetSplinesUsed() <= sIndex) {
        throw std::runtime_error("GetSplineValue: Spline index invalid");
    }
    // This can trigger a *slow* copy of the spline values from the GPU to the
    // CPU.
    return fSplineValue->hostPtr() + sIndex;
}
#endif

// Define CACHE_DEBUG to get lots of output from the host
#undef CACHE_DEBUG

#include "CacheAtomicMult.h"

namespace {
    // Interpolate one point.  This is the only place that changes when the
    // interpolation method changes.  This accepts a normalized "x" value, and
    // an array of control points with "dim" entries..  The control points
    // will be at (0, 1.0, 2.0, ... , dim-1).  The input variable "x" must be
    // a "floating point" index. If the index "x" is out of range, then this
    // turns into a linear extrapolation of the boundary points (try to avoid
    // that).
    //
    // Example: If the control points have dim of 5, the index "x" must be
    // greater than zero, and less than 5.  Assuming linear interpolation, an
    // input value of 2.1 results in the linear interpolation between element
    // [2] and element [3], or "(1.0-0.1)*p[2] + 0.1*p[3])".
    HEMI_DEV_CALLABLE_INLINE
    float HEMIInterp(double x, const float* points, int dim) {
#undef USE_LINEAR_INTERPOLATION
#ifdef USE_LINEAR_INTERPOLATION
#error Linear interpolation requested
        // Get the integer part and bound it to a valid index.
        int ix = x;
        if (ix < 0) ix = 0;
        if (ix > dim-2) ix = dim-2;

        // Get the remainder for the "index".  If the input index is less than
        // zero or greater than kPointSize-1, this is the distance from the
        // boundary.
        float fx = x-ix;
        float ffx = 1.0-fx;

        float v = ffx*points[ix] + fx*points[ix+1];
#endif
#define USE_SPLINE_INTERPOLATION
#ifdef USE_SPLINE_INTERPOLATION
        // Interpolate between p2 and p3
        // ix-2 ix-1 ix
        // p0   p1   p2---p3   p4   p5
        //   d10  d21  d32  d43  d54
        // m0| |m1| |m2| |m3| |m4| |m5
        //  a0 ||a1 ||a2 ||a3 ||a4 ||a5
        //     b0   b1   b2   b3   b4

        // Get the integer part
        int ix = x;

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
        float p2 = points[d32_0];
        float p3 = points[d32_1];

        // Get the remainder for the "index".  If the input index is less than
        // zero or greater than kPointSize-1, this is the distance from the
        // boundary.
        float fx = x-d32_0;
        float fxx = fx*fx;
        float fxxx = fx*fxx;

        // Get the values of the deltas
        // float d10 = points[IX(i,d10_1)] - points[IX(i,d10_0)];
        float d21 = points[d21_1] - points[d21_0];
        float d32 = p3-p2;
        float d43 = points[d43_1] - points[d43_0];

        // Find the raw slopes at each point
        // float m1 = 0.5*(d10+d21);
        float m2 = 0.5*(d21+d32);
        float m3 = 0.5*(d32+d43);

#ifdef COMPACT_SPLINE_MONOTONIC
        #warning Using a MONOTONIC spline
        float d54 = points[d54_1] - points[d54_0];
        float m4 = 0.5*(d43+d54);

        // Deal with cusp points and flat areas.
        // if (d21*d10 < 0.0) m1 = 0.0;
        if (d32*d21 <= 0.0) m2 = 0.0;
        if (d43*d32 <= 0.0) m3 = 0.0;
        if (d54*d43 <= 0.0) m4 = 0.0;

        // Find the alphas and betas
        // float a0 = (d10>0) ? m0/d21: 0;
        // float b0 = (d10>0) ? m1/d21: 0;
        // float a1 = (d21>0) ? m1/d21: 0;
        float b1 = (d21>0) ? m2/d21: 0;
        float a2 = (d32>0) ? m2/d32: 0;
        float b2 = (d32>0) ? m3/d32: 0;
        float a3 = (d43>0) ? m3/d43: 0;
        float b3 = (d43>0) ? m4/d43: 0;
        // float a4 = (d54>0) ? m4/d54: 0;
        // float b4 = (d54>0) ? m5/d54: 0;

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

#endif
        return v;
    }

#define PRINT_STEP 3

    // A function to be used as the kernel on either the CPU or GPU.  This
    // must be valid CUDA coda.
    HEMI_KERNEL_FUNCTION(HEMISplinesKernel,
                         double* results,
#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning Using SLOW VALIDATION in Cache::Weight::CompactSpline::HEMISplinesKernel
                         // inputs/output for validation
                         double* splineValues,
#endif
                         const double* params,
                         const double* lowerClamp,
                         const double* upperClamp,
                         const float* knots,
                         const int* rIndex,
                         const short* pIndex,
                         const int* sIndex,
                         const int NP) {
        for (int i : hemi::grid_stride_range(0,NP)) {
            const int id0 = sIndex[i];
            const int id1 = sIndex[i+1];
            double x = params[pIndex[i]];
            x = (x-knots[id0])*knots[id0+1];
            x = HEMIInterp(x, &knots[id0+2], id1-id0-2);
            float lc = lowerClamp[pIndex[i]];
            if (x < lc) x = lc;
            float uc = upperClamp[pIndex[i]];
            if (x > uc) x = uc;
#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning Using SLOW VALIDATION in Cache::Weight::CompactSpline::HEMISplinesKernel
            splineValues[i] = x;
#endif
            CacheAtomicMult(&results[rIndex[i]], x);
#ifndef HEMI_DEV_CODE
#ifdef CACHE_DEBUG
            if (rIndex[i] < PRINT_STEP) {
                LOGGER << "Splines kernel " << i
                       << " iEvt " << rIndex[i]
                       << " iPar " << pIndex[i]
                       << " = " << params[pIndex[i]]
                       << " --> " << x
                       << std::endl;
            }
#endif
#endif
        }
    }
}

bool Cache::Weight::CompactSpline::Apply() {
    if (GetSplinesUsed() < 1) return false;
    HEMISplinesKernel splinesKernel;
#ifdef FORCE_HOST_KERNEL
    splinesKernel(fWeights.hostPtr(),
#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning Using SLOW VALIDATION in Cache::Weight::CompactSpline::Apply
                  fSplineValue->hostPtr(),
#endif
                  fParameters.hostPtr(),
                  fLowerClamp.hostPtr(),
                  fUpperClamp.hostPtr(),
                  fSplineKnots->hostPtr(),
                  fSplineResult->hostPtr(),
                  fSplineParameter->hostPtr(),
                  fSplineIndex->hostPtr(),
                  GetSplinesUsed()
        );
#else
    hemi::launch(splinesKernel,
                 fWeights.writeOnlyPtr(),
#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning Using SLOW VALIDATION in Cache::Weight::CompactSpline::Apply
                 fSplineValue->writeOnlyPtr(),
#endif
                 fParameters.readOnlyPtr(),
                 fLowerClamp.readOnlyPtr(),
                 fUpperClamp.readOnlyPtr(),
                 fSplineKnots->readOnlyPtr(),
                 fSplineResult->readOnlyPtr(),
                 fSplineParameter->readOnlyPtr(),
                 fSplineIndex->readOnlyPtr(),
                 GetSplinesUsed()
        );
#endif

#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning Using SLOW VALIDATION and copying spine values
    fSplineValue->hostPtr();
#endif

    return true;
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
