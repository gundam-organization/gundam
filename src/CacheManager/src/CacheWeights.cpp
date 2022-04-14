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

#include "Logger.h"
LoggerInit([](){
  Logger::setUserHeaderStr("[Cache]");
})

// The constructor
Cache::Weights::Weights(
    Cache::Parameters::Values& parameters,
    Cache::Parameters::Clamps& lowerClamp,
    Cache::Parameters::Clamps& upperClamp,
    std::size_t results)
    : fParameters(parameters), fLowerClamp(lowerClamp), fUpperClamp(upperClamp),
      fTotalBytes(0), fResultCount(results) {
    if (fResultCount<1) throw std::runtime_error("No results in weight cache");

    LogInfo << "Cached Weights -- output results reserved: "
           << GetResultCount()
           << std::endl;
    fTotalBytes = 0;
    fTotalBytes += GetResultCount()*sizeof(double);   // fResults
    fTotalBytes += GetResultCount()*sizeof(double);   // fInitialValues;

    LogInfo << "Cached Weights -- approximate memory size: " << fTotalBytes/1E+9
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
        LogError << "Failed to allocate memory, so stopping" << std::endl;
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

double Cache::Weights::GetResult(int i) {
    if (i < 0) throw;
    if (GetResultCount() <= i) throw;
    fResultsValid = true;
    return fResults->hostPtr()[i];
}

double Cache::Weights::GetResultFast(int i) {
    fResultsValid = true;
    return fResults->readOnlyHostPtr()[i];
}

void Cache::Weights::SetResult(int i, double v) {
    if (i < 0) throw;
    if (GetResultCount() <= i) throw;
    fResults->hostPtr()[i] = v;
}

double* Cache::Weights::GetResultPointer(int i) {
    if (i < 0) throw;
    if (GetResultCount() <= i) throw;
    return (fResults->hostPtr() + i);
}

bool* Cache::Weights::GetResultValidPointer() {
    return &fResultsValid;
}

double  Cache::Weights::GetInitialValue(int i) {
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
                std::cout << "Set kernel result " << i << " to " << results[i]
                          << std::endl;
            }
#endif
#endif
        }
    }
}

bool Cache::Weights::Apply() {

    HEMISetKernel setKernel;
    hemi::launch(setKernel,
                 fResults->writeOnlyPtr(),
                 fInitialValues->readOnlyPtr(),
                 GetResultCount());

    for (int i=0; i<fWeightCalculators; ++i) {
        if (!fWeightCalculator.at(i)) continue;
        fWeightCalculator.at(i)->Apply();
    }

    // Mark the results has having changed.
    fResultsValid = false;

    // Synchronization prevents the GPU from running in parallel with the CPU,
    // so it can make the whole program a little slower.  In practice, the
    // synchronization doesn't slow things down in GUNDAM.  The suspicion is
    // that it's because the CPU almost immediately uses the results, and the
    // sync prevents a small amount of mutex locking.
    // hemi::deviceSynchronize();

    // A simple way to force a copy from the device.
    // fResults->hostPtr();

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
