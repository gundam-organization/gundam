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

// The constructor
Cache::Weights::Weights(
    Cache::Parameters::Values& parameters,
    Cache::Parameters::Clamps& lowerClamp,
    Cache::Parameters::Clamps& upperClamp,
    std::size_t results)
    : fParameters(parameters), fLowerClamp(lowerClamp), fUpperClamp(upperClamp),
      fTotalBytes(0), fResultCount(results) {
    if (fResultCount<1) throw std::runtime_error("No results in weight cache");

    LOGGER << "Cached Weights: output results: "
           << GetResultCount()
           << std::endl;
    fTotalBytes += GetResultCount()*sizeof(double);   // fResults
    fTotalBytes += GetResultCount()*sizeof(double);   // fInitialValues;

    LOGGER << "Approximate Memory Size: " << fTotalBytes/1E+9
              << " GB" << std::endl;

    try {
        // Get CPU/GPU memory for the results and thier initial values.  The
        // results are copied every time, so pin the CPU memory into the page
        // set.  The initial values are seldom changed, so they are not
        // pinned.
        fResults.reset(new hemi::Array<double>(GetResultCount(),true));
        fInitialValues.reset(new hemi::Array<double>(GetResultCount(),false));

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

    for (int i=0; i<fWeightCalculators; ++i) {
        if (!fWeightCalculator.at(i)) continue;
        fWeightCalculator.at(i)->Apply();
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
