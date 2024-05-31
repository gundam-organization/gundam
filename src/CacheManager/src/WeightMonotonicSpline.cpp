#include "CacheWeights.h"
#include "WeightBase.h"
#include "WeightMonotonicSpline.h"

#include <algorithm>
#include <iostream>
#include <exception>
#include <limits>
#include <cmath>

#include <hemi/hemi_error.h>
#include <hemi/launch.h>
#include <hemi/grid_stride_range.h>

#include "Logger.h"
LoggerInit([]{
  Logger::setUserHeaderStr("[Cache::Weight::MonotonicSpline]");
});

// The constructor
Cache::Weight::MonotonicSpline::MonotonicSpline(
    Cache::Weights::Results& weights,
    Cache::Parameters::Values& parameters,
    Cache::Parameters::Clamps& lowerClamps,
    Cache::Parameters::Clamps& upperClamps,
    std::size_t splines, std::size_t knots,
    std::string spaceOption)
    : Cache::Weight::Base("compactSpline",weights,parameters),
      fLowerClamp(lowerClamps), fUpperClamp(upperClamps),
      fSplinesReserved(splines), fSplinesUsed(0),
      fSplineSpaceReserved(knots), fSplineSpaceUsed(0) {

    LogInfo << "Reserved " << GetName() << " Splines: "
           << GetSplinesReserved() << std::endl;
    if (GetSplinesReserved() < 1) return;

    fTotalBytes += GetSplinesReserved()*sizeof(int);      // fSplineResult
    fTotalBytes += GetSplinesReserved()*sizeof(short);    // fSplineParameter
    fTotalBytes += (1+GetSplinesReserved())*sizeof(int);  // fSplineIndex

    if (spaceOption == "points") {
        fSplineSpaceReserved = 2*fSplinesReserved + fSplineSpaceReserved;
    }
    else {
        LogThrowIf(spaceOption != "space",
                   "Invalid space option for compact splines");
    }

    LogInfo << "Reserved " << GetName()
            << " Spline Knots: " << GetSplineSpaceReserved()
            << std::endl;
    fTotalBytes += GetSplineSpaceReserved()*sizeof(WEIGHT_BUFFER_FLOAT);  // fSplineKnots


    LogInfo << "Approximate Memory Size for " << GetName()
            << ": " << fTotalBytes/1E+9
            << " GB" << std::endl;

    try {
        // Get the CPU/GPU memory for the spline index tables.  These are
        // copied once during initialization so do not pin the CPU memory into
        // the page set.
        fSplineResult.reset(new hemi::Array<int>(GetSplinesReserved(),false));
        LogThrowIf(not fSplineResult, "Bad SplineResult alloc");
        fSplineParameter.reset(
            new hemi::Array<short>(GetSplinesReserved(),false));
        LogThrowIf(not fSplineParameter, "Bad SplineParameter alloc");
        fSplineIndex.reset(new hemi::Array<int>(1+GetSplinesReserved(),false));
        LogThrowIf(not fSplineIndex, "Bad SplineIndex alloc");

        // Get the CPU/GPU memory for the spline knots.  This is copied once
        // during initialization so do not pin the CPU memory into the page
        // set.
        fSplineSpace.reset(
            new hemi::Array<WEIGHT_BUFFER_FLOAT>(GetSplineSpaceReserved(),false));
        LogThrowIf(not fSplineSpace, "Bad SplineSpace alloc");
    }
    catch (...) {
        LogError << "Uncaught exception in WeightGraph" << std::endl;
        LogThrow("WeightGraph -- uncaught exception");
    }

    // Initialize the caches.  Don't try to zero everything since the
    // caches can be huge.
    Reset();
    fSplineIndex->hostPtr()[0] = 0;
}

// The destructor
Cache::Weight::MonotonicSpline::~MonotonicSpline() {}

int Cache::Weight::MonotonicSpline::FindPoints(const TSpline3* s) {
    return s->GetNp();
}

void Cache::Weight::MonotonicSpline::AddSpline(int resIndex,
                                               int parIndex,
                                               const std::vector<double>& splineData) {
    if (resIndex < 0) {
        LogError << "Invalid result index"
               << std::endl;
        LogThrow("Negative result index");
    }
    if (fWeights.size() <= resIndex) {
        LogError << "Invalid result index"
               << std::endl;
        LogThrow("Result index out of bounds");
    }
    if (parIndex < 0) {
        LogError << "Invalid parameter index"
               << std::endl;
        LogThrow("Negative parameter index");
    }
    if (fParameters.size() <= parIndex) {
        LogError << "Invalid parameter index: " << parIndex
                 << " out of " << fParameters.size()
                 << std::endl;
        LogThrow("Parameter index out of bounds");
    }
    if (splineData.size() < 5) {
        LogError << "Insufficient points in spline: " << splineData.size()
               << std::endl;
        LogThrow("Invalid number of spline points");
    }
    int newIndex = fSplinesUsed++;
    if (fSplinesUsed > fSplinesReserved) {
        LogError << "Not enough space reserved for splines"
                 << " Reserved: " << fSplinesReserved
                 << " Used: " << fSplinesUsed
                 << std::endl;
        LogThrow("Not enough space reserved for splines");
    }
    fSplineResult->hostPtr()[newIndex] = resIndex;
    fSplineParameter->hostPtr()[newIndex] = parIndex;
    if (fSplineIndex->hostPtr()[newIndex] != fSplineSpaceUsed) {
        LogError << "Last spline knot index should be at old end of splines"
                  << std::endl;
        LogThrow("Problem with control indices");
    }
    int knotIndex = fSplineSpaceUsed;
    fSplineSpaceUsed += splineData.size();
    if (fSplineSpaceUsed > fSplineSpaceReserved) {
        LogError << "Not enough space reserved for spline knots"
                 << " Reserved: " << fSplineSpaceReserved
                 << " Used: " << fSplineSpaceUsed
                 << std::endl;
        LogThrow("Not enough space reserved for spline knots");
    }
    fSplineIndex->hostPtr()[newIndex+1] = fSplineSpaceUsed;
    for (std::size_t i = 0; i<splineData.size(); ++i) {
        fSplineSpace->hostPtr()[knotIndex+i] = splineData.at(i);
    }

}

void Cache::Weight::MonotonicSpline::SetSplineKnot(
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
    int knotIndex = fSplineIndex->hostPtr()[sIndex] + 2 + kIndex;
    if (fSplineIndex->hostPtr()[sIndex+1] <= knotIndex) {
        LogError << "Requested control point index is two large"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    fSplineSpace->hostPtr()[knotIndex] = value;
}

int Cache::Weight::MonotonicSpline::GetSplineParameterIndex(int sIndex) {
    if (sIndex < 0) {
        LogThrow("Spline index invalid");
    }
    if (GetSplinesUsed() <= sIndex) {
        LogThrow("Spline index invalid");
    }
    return fSplineParameter->hostPtr()[sIndex];
}

double Cache::Weight::MonotonicSpline::GetSplineParameter(int sIndex) {
    int i = GetSplineParameterIndex(sIndex);
    if (i<0) {
        LogThrow("Spline parameter index out of bounds");
    }
    if (fParameters.size() <= i) {
        LogThrow("Spline parameter index out of bounds");
    }
    return fParameters.hostPtr()[i];
}

int Cache::Weight::MonotonicSpline::GetSplineKnotCount(int sIndex) {
    if (sIndex < 0) {
        LogThrow("Spline index invalid");
    }
    if (GetSplinesUsed() <= sIndex) {
        LogThrow("Spline index invalid");
    }
    return fSplineIndex->hostPtr()[sIndex+1]-fSplineIndex->hostPtr()[sIndex]-2;
}

double Cache::Weight::MonotonicSpline::GetSplineLowerBound(int sIndex) {
    if (sIndex < 0) {
        LogThrow("Spline index invalid");
    }
    if (GetSplinesUsed() <= sIndex) {
        LogThrow("Spline index invalid");
    }
    int knotsIndex = fSplineIndex->hostPtr()[sIndex];
    return fSplineSpace->hostPtr()[knotsIndex];
}

double Cache::Weight::MonotonicSpline::GetSplineUpperBound(int sIndex) {
    if (sIndex < 0) {
        LogThrow("Spline index invalid");
    }
    if (GetSplinesUsed() <= sIndex) {
        LogThrow("Spline index invalid");
    }
    int knotCount = GetSplineKnotCount(sIndex);
    double lower = GetSplineLowerBound(sIndex);
    int knotsIndex = fSplineIndex->hostPtr()[sIndex];
    double step = fSplineSpace->hostPtr()[knotsIndex+1];
    return lower + (knotCount-1)/step;
}

double Cache::Weight::MonotonicSpline::GetSplineLowerClamp(int sIndex) {
    int i = GetSplineParameterIndex(sIndex);
    if (i<0) {
        LogThrow("Spline lower clamp index out of bounds");
    }
    if (fLowerClamp.size() <= i) {
        LogThrow("Spline lower clamp index out of bounds");
    }
    return fLowerClamp.hostPtr()[i];
}

double Cache::Weight::MonotonicSpline::GetSplineUpperClamp(int sIndex) {
    int i = GetSplineParameterIndex(sIndex);
    if (i<0) {
        LogThrow("Spline upper clamp index out of bounds");
    }
    if (fUpperClamp.size() <= i) {
        LogThrow("Spline upper clamp index out of bounds");
    }
    return fUpperClamp.hostPtr()[i];
}

double Cache::Weight::MonotonicSpline::GetSplineKnot(int sIndex, int knot) {
    if (sIndex < 0) {
        LogThrow("Spline index invalid");
    }
    if (GetSplinesUsed() <= sIndex) {
        LogThrow("Spline index invalid");
    }
    int knotsIndex = fSplineIndex->hostPtr()[sIndex];
    int count = GetSplineKnotCount(sIndex);
    if (knot < 0) {
        LogThrow("Knot index invalid");
    }
    if (count <= knot) {
        LogThrow("Knot index invalid");
    }
    return fSplineSpace->hostPtr()[knotsIndex+2+knot];
}

#include "CacheAtomicMult.h"
#include "CalculateMonotonicSpline.h"

// Define CACHE_DEBUG to get lots of output from the host
#undef CACHE_DEBUG
#define PRINT_STEP 3

namespace {
    // A function to be used as the kernel on either the CPU or GPU.  This
    // must be valid CUDA coda.
    HEMI_KERNEL_FUNCTION(HEMISplinesKernel,
                         double* results,
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
            const int dim = id1-id0-2;
            const double x = params[pIndex[i]];
            const double lClamp = lowerClamp[pIndex[i]];
            const double uClamp = upperClamp[pIndex[i]];

            double v = CalculateMonotonicSpline(x, lClamp,uClamp,
                                                &knots[id0],dim);
            CacheAtomicMult(&results[rIndex[i]], v);
#ifndef HEMI_DEV_CODE
#ifdef CACHE_DEBUG
            if (rIndex[i] < PRINT_STEP) {
                std::cout << "Splines kernel " << i
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

void Cache::Weight::MonotonicSpline::Reset() {
    // Use the parent reset.
    Cache::Weight::Base::Reset();
    // Reset this class
    fSplinesUsed = 0;
    fSplineSpaceUsed = 0;
}

bool Cache::Weight::MonotonicSpline::Apply() {
    if (GetSplinesUsed() < 1) return false;

    HEMISplinesKernel splinesKernel;
    hemi::launch(splinesKernel,
                 fWeights.writeOnlyPtr(),
                 fParameters.readOnlyPtr(),
                 fLowerClamp.readOnlyPtr(),
                 fUpperClamp.readOnlyPtr(),
                 fSplineSpace->readOnlyPtr(),
                 fSplineResult->readOnlyPtr(),
                 fSplineParameter->readOnlyPtr(),
                 fSplineIndex->readOnlyPtr(),
                 GetSplinesUsed()
        );

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
