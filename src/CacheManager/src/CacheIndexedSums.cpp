#include "CacheIndexedSums.h"
#include "CacheWeights.h"

#include <iostream>
#include <exception>
#include <cmath>
#include <memory>

#include <hemi/hemi_error.h>
#include <hemi/launch.h>
#include <hemi/grid_stride_range.h>

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[Cache::IndexedSums]");
});

// The constructor
Cache::IndexedSums::IndexedSums(Cache::Weights::Results& inputs,
                                std::size_t bins)
    : fEventWeights(inputs) {
    if (inputs.size()<1) throw std::runtime_error("No bins to sum");
    if (bins<1) throw std::runtime_error("No bins to sum");

    LogInfo << "Cached IndexedSums -- bins reserved: "
           << bins
           << std::endl;
    fTotalBytes += bins*sizeof(double);                   // fSums
    fTotalBytes += fEventWeights.size()*sizeof(short);   // fIndexes;

    LogInfo << "Cached IndexedSums -- approximate memory size: "
            << double(fTotalBytes)/1E+6
            << " MB" << std::endl;

    try {
        // Get CPU/GPU memory for the results and thier initial values.  The
        // results are copied every time, so pin the CPU memory into the page
        // set.  The initial values are seldom changed, so they are not
        // pinned.
        fSums = std::make_unique<hemi::Array<double>>(bins,true);
        fIndexes = std::make_unique<hemi::Array<short>>(fEventWeights.size(),false);

    }
    catch (std::bad_alloc&) {
        LogError << "Failed to allocate memory, so stopping" << std::endl;
        throw std::runtime_error("Not enough memory available");
    }

    // Initialize the caches.  Don't try to zero everything since the
    // caches can be huge.
    std::fill(fSums->hostPtr(),
              fSums->hostPtr() + fSums->size(),
              0.0);
}

// The destructor
Cache::IndexedSums::~IndexedSums() = default;

void Cache::IndexedSums::SetEventIndex(int event, int bin) {
    if (event < 0) throw;
    if (fEventWeights.size() <= event) throw;
    if (bin < 0) throw;
    if (fSums->size() <= bin) throw;
    fIndexes->hostPtr()[event] = bin;
}

double Cache::IndexedSums::GetSum(int i) {
    if (i < 0) throw;
    if (fSums->size() <= i) throw;
    // This odd ordering is to make sure the thread-safe hostPtr update
    // finishes before the sum is set to be valid.  The use of isfinite is to
    // make sure that the optimizer doesn't reorder the statements.
    double value = fSums->hostPtr()[i];
    if (std::isfinite(value)) fSumsValid = true;
    return value;
}

const double* Cache::IndexedSums::GetSumsPointer() {
    return fSums->hostPtr();
}

bool* Cache::IndexedSums::GetSumsValidPointer() {
    return &fSumsValid;
}

// Define CACHE_DEBUG to get lots of output from the host
#undef CACHE_DEBUG

#include "CacheAtomicAdd.h"

namespace {
    // A function to be used as the kernen on a CPU or GPU.  This must be
    // valid CUDA.  This sets all of the results to a fixed value.
    HEMI_KERNEL_FUNCTION(HEMIResetKernel,
                         double* sums,
                         const double value,
                         const int NP) {
        for (int i : hemi::grid_stride_range(0,NP)) {
            sums[i] = value;
        }
    }

    // A function to do the sums
    HEMI_KERNEL_FUNCTION(HEMIIndexedSumKernel,
                         double* sums,
                         const double* inputs,
                         const short* indexes,
                         const int NP) {
        for (int i : hemi::grid_stride_range(0,NP)) {
#ifdef HEMI_DEV_CODE
            CacheAtomicAdd(&sums[indexes[i]],inputs[i]);
#else
            sums[indexes[i]] += inputs[i];
#endif
        }
    }

}

bool Cache::IndexedSums::Apply() {
    // Mark the results has having changed.
    fSumsValid = false;

    HEMIResetKernel resetKernel;
    hemi::launch(resetKernel,
                 fSums->writeOnlyPtr(),
                 0.0,
                 fSums->size());

    HEMIIndexedSumKernel indexedSumKernel;
    hemi::launch(indexedSumKernel,
                 fSums->writeOnlyPtr(),
                 fEventWeights.readOnlyPtr(),
                 fIndexes->readOnlyPtr(),
                 fEventWeights.size());

    // Synchronization prevents the GPU from running in parallel with the CPU,
    // so it can make the whole program a little slower.  In practice, the
    // synchronization doesn't slow things down in GUNDAM.  The suspicion is
    // that it's because the CPU almost immediately uses the results, and the
    // sync prevents a small amount of mutex locking.
    // hemi::deviceSynchronize();

    // A simple way to force a copy from the device.
    // fSums->hostPtr();

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
