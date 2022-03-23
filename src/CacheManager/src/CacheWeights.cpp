#include "CacheWeights.h"
#include "WeightNormalization.h"
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

Cache::Weights* Cache::Weights::fSingleton = nullptr;

// The constructor
Cache::Weights::Weights(
    std::size_t results, std::size_t parameters, std::size_t norms,
    std::size_t splines, std::size_t knots)
    : fTotalBytes(0), fResultCount(results), fParameterCount(parameters) {
    if (fResultCount<1) throw std::runtime_error("No results in weight cache");
    if (fParameterCount<1) throw std::runtime_error("No parameters");

    LOGGER << "Cached Weights: input parameter count: "
           << GetParameterCount()
           << std::endl;
    fTotalBytes += GetParameterCount()*sizeof(double); // fParameters
    fTotalBytes += GetParameterCount()*sizeof(double);  // fLowerClamp
    fTotalBytes += GetParameterCount()*sizeof(double);  // fUpperclamp

    LOGGER << "Cached Weights: output results: "
           << GetResultCount()
           << std::endl;
    fTotalBytes += GetResultCount()*sizeof(double);   // fResults
    fTotalBytes += GetResultCount()*sizeof(double);   // fInitialValues;

    LOGGER << "Approximate Memory Size: " << fTotalBytes/1E+9
              << " GB" << std::endl;

    try {
        // The mirrors are only on the CPU, so use vectors.  Initialize with
        // lowest and highest floating point values.
        fLowerMirror.reset(new std::vector<double>(
                               GetParameterCount(),
                               std::numeric_limits<double>::lowest()));
        fUpperMirror.reset(new std::vector<double>(
                               GetParameterCount(),
                               std::numeric_limits<double>::max()));

        // Get CPU/GPU memory for the parameter values.  The mirroring is done
        // to every entry, so its also done on the GPU.  The parameters are
        // copied every iteration, so pin the CPU memory into the page set.
        fParameters.reset(new hemi::Array<double>(GetParameterCount()));
        fLowerClamp.reset(new hemi::Array<double>(GetParameterCount(),false));
        fUpperClamp.reset(new hemi::Array<double>(GetParameterCount(),false));

        // Get CPU/GPU memory for the results and thier initial values.  The
        // results are copied every time, so pin the CPU memory into the page
        // set.  The initial values are seldom changed, so they are not
        // pinned.
        fResults.reset(new hemi::Array<double>(GetResultCount(),true));
        fInitialValues.reset(new hemi::Array<double>(GetResultCount(),false));

        fWeights[0].reset(new Cache::Weight::Normalization(
                              *fResults,*fParameters,
                              norms));
        fWeights[1].reset(new Cache::Weight::CompactSpline(
                              *fResults,*fParameters,*fLowerClamp,*fUpperClamp,
                              splines, knots));
    }
    catch (std::bad_alloc&) {
        LOGGER << "Failed to allocate memory, so stopping" << std::endl;
        throw std::runtime_error("Not enough memory available");
    }

    // Initialize the caches.  Don't try to zero everything since the
    // caches can be huge.
    std::fill(fInitialValues->hostPtr(),
              fInitialValues->hostPtr() + fInitialValues->size(),
              1.0);
    std::fill(fLowerClamp->hostPtr(),
              fLowerClamp->hostPtr() + GetParameterCount(),
              std::numeric_limits<double>::lowest());
    std::fill(fUpperClamp->hostPtr(),
              fUpperClamp->hostPtr() + GetParameterCount(),
              std::numeric_limits<double>::max());

}

// The destructor
Cache::Weights::~Weights() {}

double Cache::Weights::GetResult(int i) const {
    if (i < 0) throw;
    if (GetResultCount() <= i) throw;
    return fResults->hostPtr()[i];
}

void Cache::Weights::SetResult(int i, double v) {
    if (i < 0) throw;
    if (GetResultCount() <= i) throw;
    fResults->hostPtr()[i] = v;
}

double* Cache::Weights::GetResultPointer(int i) const {
    if (i < 0) throw;
    if (GetResultCount() <= i) throw;
    return (fResults->hostPtr() + i);
}

double  Cache::Weights::GetInitialValue(int i) const {
    if (i < 0) throw;
    if (GetResultCount() <= i) throw;
    return fInitialValues->hostPtr()[i];
}

void Cache::Weights::SetInitialValue(int i, double v) {
    if (i < 0) throw;
    if (GetResultCount() <= i) throw;
    fInitialValues->hostPtr()[i] = v;
}

double Cache::Weights::GetParameter(int parIdx) const {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    return fParameters->hostPtr()[parIdx];
}

void Cache::Weights::SetParameter(int parIdx, double value) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    fParameters->hostPtr()[parIdx] = value;
}

double Cache::Weights::GetLowerMirror(int parIdx) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    return fLowerMirror->at(parIdx);
}

void Cache::Weights::SetLowerMirror(int parIdx, double value) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    fLowerMirror->at(parIdx) = value;
}

double Cache::Weights::GetUpperMirror(int parIdx) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    return fUpperMirror->at(parIdx);
}

void Cache::Weights::SetUpperMirror(int parIdx, double value) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    fUpperMirror->at(parIdx) = value;
}

double Cache::Weights::GetLowerClamp(int parIdx) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    return fLowerClamp->hostPtr()[parIdx];
}

void Cache::Weights::SetLowerClamp(int parIdx, double value) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    fLowerClamp->hostPtr()[parIdx] = value;
}

double Cache::Weights::GetUpperClamp(int parIdx) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    return fUpperClamp->hostPtr()[parIdx];
}

void Cache::Weights::SetUpperClamp(int parIdx, double value) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    fUpperClamp->hostPtr()[parIdx] = value;
}

// Reserve space for another normalization parameter.
int Cache::Weights::ReserveNorm(int resIndex, int parIndex) {
    return dynamic_cast<Cache::Weight::Normalization&>(*fWeights[0])
        .ReserveNorm(resIndex,parIndex);
}

// Reserve space in the internal structures for spline with uniform knots.
// The knot values will still need to be set using set spline knot.
int Cache::Weights::ReserveSpline(
    int resIndex, int parIndex, double low, double high, int points) {
    return dynamic_cast<Cache::Weight::CompactSpline&>(*fWeights[1])
        .ReserveSpline(resIndex,parIndex,low,high,points);
}

void Cache::Weights::SetSplineKnot(
    int sIndex, int kIndex, double value) {
    dynamic_cast<Cache::Weight::CompactSpline&>(*fWeights[1])
        .SetSplineKnot(sIndex,kIndex,value);
}

// A convenience function combining ReserveSpline and SetSplineKnot.
int Cache::Weights::AddSpline(
    int resIndex, int parIndex, double low, double high,
    double points[], int nPoints) {
    return dynamic_cast<Cache::Weight::CompactSpline&>(*fWeights[1])
        .AddSpline(resIndex,parIndex,low,high,points,nPoints);
}

// Define CACHE_DEBUG to get lots of output from the host
#undef CACHE_DEBUG

#include "CacheAtomicMult.h"

namespace {
    // A function to be used as the kernen on a CPU or GPU.  This must be
    // valid CUDA.  This sets all of the results to a fixed value.
    HEMI_KERNEL_FUNCTION(HEMISetKernel,
                         double* results,
                         const double* values,
                         const int NP) {
        for (int i : hemi::grid_stride_range(0,NP)) {
            results[i] = values[i];
#ifndef HEMI_DEV_CODE
#ifdef CACHE_DEBUG
            if (i < PRINT_STEP) {
                LOGGER << "Set kernel result " << i << " to " << results[i]
                       << std::endl;
            }
#endif
#endif
        }
    }
}

bool Cache::Weights::Apply() {
    for (int i = 0; i<GetParameterCount(); ++i) {
        double lm = fLowerMirror->at(i);
        double um = fUpperMirror->at(i);
        double v = GetParameter(i);
        // Mirror the input value at lm and um.
        int brake = 20;
        while (v < lm || v > um) {
            if (v < lm) v = lm + (lm - v);
            if (v > um) v = um - (v - um);
            if (--brake < 1) throw;
        }
        SetParameter(i,v);
    }

    HEMISetKernel setKernel;
#ifdef FORCE_HOST_KERNEL
    setKernel(   fResults->hostPtr(),
                 fInitialValues->hostPtr(),
                 GetResultCount());
#else
    hemi::launch(setKernel,
                 fResults->writeOnlyPtr(),
                 fInitialValues->readOnlyPtr(),
                 GetResultCount());
#endif

    for (int i=0; i<fWeights.size(); ++i) {
        if (!fWeights[i]) continue;
        fWeights[i]->Apply();
    }

    // A simple way to copy from the device.  This needs to be done since
    // other places using the values are referencing the contents of the host
    // array by address, and that won't trigger the copy.  The copy also isn't
    // thread safe.
    fResults->hostPtr();

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
