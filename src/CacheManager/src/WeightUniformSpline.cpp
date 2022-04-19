#include "CacheWeights.h"
#include "WeightBase.h"
#include "WeightUniformSpline.h"

#include <algorithm>
#include <iostream>
#include <exception>
#include <limits>
#include <cmath>

#include <hemi/hemi_error.h>
#include <hemi/launch.h>
#include <hemi/grid_stride_range.h>

#include "Logger.h"
LoggerInit([](){
  Logger::setUserHeaderStr("[Cache]");
})

// The constructor
Cache::Weight::UniformSpline::UniformSpline(
    Cache::Weights::Results& weights,
    Cache::Parameters::Values& parameters,
    Cache::Parameters::Clamps& lowerClamps,
    Cache::Parameters::Clamps& upperClamps,
    std::size_t splines, std::size_t knots)
    : Cache::Weight::Base("uniformSpline",weights,parameters),
      fLowerClamp(lowerClamps), fUpperClamp(upperClamps),
      fSplinesReserved(splines), fSplinesUsed(0),
      fSplineKnotsReserved(knots), fSplineKnotsUsed(0) {

    LogInfo << "Reserved " << GetName() << " Splines: "
           << GetSplinesReserved() << std::endl;
    if (GetSplinesReserved() < 1) return;

    fTotalBytes += GetSplinesReserved()*sizeof(int);      // fSplineResult
    fTotalBytes += GetSplinesReserved()*sizeof(short);    // fSplineParameter
    fTotalBytes += (1+GetSplinesReserved())*sizeof(int);  // fSplineIndex

    fSplineKnotsReserved = 2*fSplinesReserved + 2*fSplineKnotsReserved;

#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning Using SLOW VALIDATION in Cache::Weight::UniformSpline::UniformSpline
        // Add validation code for the spline calculation.  This can be rather
        // slow, so do not use if it is not required.
    fTotalBytes += GetSplinesReserved()*sizeof(double);
#endif

    LogInfo << "Reserved " << GetName()
            << " Spline Knots: " << GetSplineKnotsReserved()
            << std::endl;
    fTotalBytes += GetSplineKnotsReserved()*sizeof(WEIGHT_BUFFER_FLOAT);  // fSpineKnots

    LogInfo << "Approximate Memory Size for " << GetName()
            << ": " << fTotalBytes/1E+9
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
#warning Using SLOW VALIDATION in Cache::Weight::UniformSpline::UniformSpline
        // Add validation code for the spline calculation.  This can be rather
        // slow, so do not use if it is not required.
        fSplineValue.reset(new hemi::Array<double>(GetSplinesReserved(),true));
#endif

        // Get the CPU/GPU memory for the spline knots.  This is copied once
        // during initialization so do not pin the CPU memory into the page
        // set.
        fSplineKnots.reset(
            new hemi::Array<WEIGHT_BUFFER_FLOAT>(GetSplineKnotsReserved(),false));
    }
    catch (std::bad_alloc&) {
        LogError << "Failed to allocate memory, so stopping" << std::endl;
        throw std::runtime_error("Not enough memory available");
    }

    // Initialize the caches.  Don't try to zero everything since the
    // caches can be huge.
    fSplineIndex->hostPtr()[0] = 0;
}

// The destructor
Cache::Weight::UniformSpline::~UniformSpline() {}

int Cache::Weight::UniformSpline::FindPoints(const TSpline3* s) {
    return s->GetNp();
}

void Cache::Weight::UniformSpline::AddSpline(int resultIndex,
                                             int parIndex,
                                             SplineDial* sDial) {
    const TSpline3* s = sDial->getSplinePtr();
    int NP = Cache::Weight::UniformSpline::FindPoints(s);
    double xMin = s->GetXmin();
    double xMax = s->GetXmax();
    int sIndex = ReserveSpline(resultIndex,parIndex,xMin,xMax,NP);
#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning Using SLOW VALIDATION in Cache::Weight::UniformSpline::AddSpline
    sDial->setCacheManagerName(GetName());
    sDial->setCacheManagerValuePointer(GetCachePointer(sIndex));
#endif
    for (int i=0; i<NP; ++i) {
        double x = xMin + i*(xMax-xMin)/(NP-1);
        double y = s->Eval(x);
        double m = s->Derivative(x);
        SetSplineKnot(sIndex,i,y,m);
    }
}

// Reserve space in the internal structures for spline with uniform knots.
// The knot values will still need to be set using set spline knot.
int Cache::Weight::UniformSpline::ReserveSpline(
    int resIndex, int parIndex, double low, double high, int points) {
    if (resIndex < 0) {
        LogError << "Invalid result index"
               << std::endl;
        throw std::runtime_error("Negative result index");
    }
    if (fWeights.size() <= resIndex) {
        LogError << "Invalid result index"
               << std::endl;
        throw std::runtime_error("Result index out of bounds");
    }
    if (parIndex < 0) {
        LogError << "Invalid parameter index"
               << std::endl;
        throw std::runtime_error("Negative parameter index");
    }
    if (fParameters.size() <= parIndex) {
        LogError << "Invalid parameter index"
               << std::endl;
        throw std::runtime_error("Parameter index out of bounds");
    }
    if (high <= low) {
        LogError << "Invalid spline bounds"
               << std::endl;
        throw std::runtime_error("Invalid spline bounds");
    }
    if (points < 3) {
        LogError << "Insufficient points in spline"
               << std::endl;
        throw std::runtime_error("Invalid number of spline points");
    }
    int newIndex = fSplinesUsed++;
    if (fSplinesUsed > fSplinesReserved) {
        LogError << "Not enough space reserved for splines"
                  << std::endl;
        throw std::runtime_error("Not enough space reserved for splines");
    }
    fSplineResult->hostPtr()[newIndex] = resIndex;
    fSplineParameter->hostPtr()[newIndex] = parIndex;
    if (fSplineIndex->hostPtr()[newIndex] != fSplineKnotsUsed) {
        LogError << "Last spline knot index should be at old end of splines"
                  << std::endl;
        throw std::runtime_error("Problem with control indices");
    }
    int knotIndex = fSplineKnotsUsed;
    fSplineKnotsUsed += 2; // Space for the upper and lower bound
    fSplineKnotsUsed += 2*points; // Space for the knots.
    if (fSplineKnotsUsed > fSplineKnotsReserved) {
        LogError << "Not enough space reserved for spline knots"
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

void Cache::Weight::UniformSpline::SetSplineKnot(
    int sIndex, int kIndex, double value, double slope) {
    SetSplineKnotValue(sIndex,kIndex,value);
    SetSplineKnotSlope(sIndex,kIndex,slope);
}

void Cache::Weight::UniformSpline::SetSplineKnotValue(
    int sIndex, int kIndex, double value) {
    if (sIndex < 0) {
        LogError << "Requested spline index is negative"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    if (GetSplinesUsed() <= sIndex) {
        LogError << "Requested spline index is to large"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    if (kIndex < 0) {
        LogError << "Requested control point index is negative"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    int knotIndex = fSplineIndex->hostPtr()[sIndex] + 2 + 2*kIndex;
    if (fSplineIndex->hostPtr()[sIndex+1] <= knotIndex) {
        LogError << "Requested control point index is two large"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    fSplineKnots->hostPtr()[knotIndex] = value;
}

void Cache::Weight::UniformSpline::SetSplineKnotSlope(
    int sIndex, int kIndex, double slope) {
    if (sIndex < 0) {
        LogError << "Requested spline index is negative"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    if (GetSplinesUsed() <= sIndex) {
        LogError << "Requested spline index is to large"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    if (kIndex < 0) {
        LogError << "Requested control point index is negative"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    int knotIndex = fSplineIndex->hostPtr()[sIndex] + 2 + 2*kIndex;
    if (fSplineIndex->hostPtr()[sIndex+1] <= knotIndex) {
        LogError << "Requested control point index is two large"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    fSplineKnots->hostPtr()[knotIndex+1] = slope;
}

int Cache::Weight::UniformSpline::GetSplineParameterIndex(int sIndex) {
    if (sIndex < 0) {
        throw std::runtime_error("Spline index invalid");
    }
    if (GetSplinesUsed() <= sIndex) {
        throw std::runtime_error("Spline index invalid");
    }
    return fSplineParameter->hostPtr()[sIndex];
}

double Cache::Weight::UniformSpline::GetSplineParameter(int sIndex) {
    int i = GetSplineParameterIndex(sIndex);
    if (i<0) {
        throw std::runtime_error("Spine parameter index out of bounds");
    }
    if (fParameters.size() <= i) {
        throw std::runtime_error("Spine parameter index out of bounds");
    }
    return fParameters.hostPtr()[i];
}

int Cache::Weight::UniformSpline::GetSplineKnotCount(int sIndex) {
    if (sIndex < 0) {
        throw std::runtime_error("Spline index invalid");
    }
    if (GetSplinesUsed() <= sIndex) {
        throw std::runtime_error("Spline index invalid");
    }
    int k = fSplineIndex->hostPtr()[sIndex+1]-fSplineIndex->hostPtr()[sIndex]-2;
    return k/2;
}

double Cache::Weight::UniformSpline::GetSplineLowerBound(int sIndex) {
    if (sIndex < 0) {
        throw std::runtime_error("Spline index invalid");
    }
    if (GetSplinesUsed() <= sIndex) {
        throw std::runtime_error("Spline index invalid");
    }
    int knotsIndex = fSplineIndex->hostPtr()[sIndex];
    return fSplineKnots->hostPtr()[knotsIndex];
}

double Cache::Weight::UniformSpline::GetSplineUpperBound(int sIndex) {
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

double Cache::Weight::UniformSpline::GetSplineLowerClamp(int sIndex) {
    int i = GetSplineParameterIndex(sIndex);
    if (i<0) {
        throw std::runtime_error("Spine lower clamp index out of bounds");
    }
    if (fLowerClamp.size() <= i) {
        throw std::runtime_error("Spine lower clamp index out of bounds");
    }
    return fLowerClamp.hostPtr()[i];
}

double Cache::Weight::UniformSpline::GetSplineUpperClamp(int sIndex) {
    int i = GetSplineParameterIndex(sIndex);
    if (i<0) {
        throw std::runtime_error("Spine upper clamp index out of bounds");
    }
    if (fUpperClamp.size() <= i) {
        throw std::runtime_error("Spine upper clamp index out of bounds");
    }
    return fUpperClamp.hostPtr()[i];
}

double Cache::Weight::UniformSpline::GetSplineKnotValue(int sIndex, int knot) {
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
    return fSplineKnots->hostPtr()[knotsIndex+2+2*knot];
}

double Cache::Weight::UniformSpline::GetSplineKnotSlope(int sIndex, int knot) {
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
    return fSplineKnots->hostPtr()[knotsIndex+2+2*knot+1];
}

////////////////////////////////////////////////////////////////////
// This section is for the validation methods.  They should mostly be
// NOOPs and should mostly not be called.

#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning Using SLOW VALIDATION in Cache::Weight::UniformSpline::GetSplineValue
// Get the intermediate spline result that is used to calculate an event
// weight.  This can trigger a copy from the GPU to CPU, and must only be
// enabled during validation.  Using this validation code also significantly
// increases the amount of GPU memory required.  In a short sentence, "Do not
// use this method."
double* Cache::Weight::UniformSpline::GetCachePointer(int sIndex) {
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
#define PRINT_STEP 3


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
    double HEMIInterp(double x, double low, double step,
                      const WEIGHT_BUFFER_FLOAT* points, int dim) {
        // Get the integer part
        const double xx = (x-low)/step;
        int ix = xx;
        if (ix<0) ix=0;
        if (2*ix+7>dim) ix = (dim-2)/2 - 2 ;

        const double fx = xx-ix;
        const double fxx = fx*fx;
        const double fxxx = fx*fxx;

        const double p1 = points[2+2*ix];
        const double m1 = points[2+2*ix+1]*step;
        const double p2 = points[2+2*ix+2];
        const double m2 = points[2+2*ix+3]*step;

        // Cubic spline with the points and slopes.
        double v = (p1*(2.0*fxxx-3.0*fxx+1.0) + m1*(fxxx-2.0*fxx+fx)
                    + p2*(3.0*fxx-2.0*fxxx) + m2*(fxxx-fxx));

        return v;
    }

    // Calculate the spline value.
    HEMI_DEV_CALLABLE_INLINE
    double HEMIUniformSpline(const double x,
                             const double lowerClamp, double upperClamp,
                             const WEIGHT_BUFFER_FLOAT* knots, const int dim) {

        const double low = knots[0];
        const double step = 1.0/knots[1];

        double v = HEMIInterp(x, low, step, knots, dim);

        if (v < lowerClamp) v = lowerClamp;
        if (v > upperClamp) v = upperClamp;

        return v;
    }

    // A function to be used as the kernel on either the CPU or GPU.  This
    // must be valid CUDA coda.
    HEMI_KERNEL_FUNCTION(HEMISplinesKernel,
                         double* results,
#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning Using SLOW VALIDATION in Cache::Weight::UniformSpline::HEMISplinesKernel
                         // inputs/output for validation
                         double* splineValues,
#endif
                         const double* params,
                         const double* lowerClamp,
                         const double* upperClamp,
                         const WEIGHT_BUFFER_FLOAT* knots,
                         const int* rIndex,
                         const short* pIndex,
                         const int* sIndex,
                         const int NP) {
        for (int i : hemi::grid_stride_range(0,NP)) {
            const int id0 = sIndex[i];
            const int id1 = sIndex[i+1];
            const int dim = id1-id0;
            const double x = params[pIndex[i]];
            const double lClamp = lowerClamp[pIndex[i]];
            const double uClamp = upperClamp[pIndex[i]];

            double v = HEMIUniformSpline(x,
                                         lClamp, uClamp,
                                         &knots[id0],dim);
#ifndef HEMI_DEV_CODE
#ifdef CACHE_DEBUG
            int dim = id1-id0-2;
            if (rIndex[i] < PRINT_STEP) {
                LogInfo << "CACHE_DEBUG: Splines kernel " << i
                       << " iEvt " << rIndex[i]
                       << " iPar " << pIndex[i]
                       << " = " << params[pIndex[i]]
                       << " m " << knots[id0] << " d "  << knots[id0+1]
                       << " (" << x << ")"
                       << " --> " << v
                       << " s: " << s
                       << " d: " << dim
                       << std::endl;
                for (int k = 0; k < dim/2; ++k) {
                    LogInfo << "CACHE_DEBUG:     " << k
                           << " x: " << knots[id0] + k*s
                           << " y: " << knots[id0+2+2*k]
                           << " m: " << knots[id0+2+2*k+1]
                           << std::endl;
                }
            }
#endif
#endif

#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning Use SLOW VALIDATION in Cache::Weight::UniformSpline::HEMISplinesKernel
            splineValues[i] = v;
#endif

            CacheAtomicMult(&results[rIndex[i]], v);
        }
    }
}

bool Cache::Weight::UniformSpline::Apply() {
    if (GetSplinesUsed() < 1) return false;

    HEMISplinesKernel splinesKernel;
    hemi::launch(splinesKernel,
                 fWeights.writeOnlyPtr(),
#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning Using SLOW VALIDATION in Cache::Weight::UniformSpline::Apply
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
