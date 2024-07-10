#include "WeightNormalization.h"

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
  Logger::setUserHeaderStr("[Cache::Weight::Normalization]");
});

// The constructor
Cache::Weight::Normalization::Normalization(
    Cache::Weights::Results& weights,
    Cache::Parameters::Values& parameters,
    std::size_t norms)
    : Cache::Weight::Base("normalization",weights,parameters),
      fNormsReserved(norms), fNormsUsed(0) {

    LogInfo << "Cached Weights: reserved Normalizations: "
           << GetNormsReserved()
           << std::endl;
    if (GetNormsReserved() < 1) return;

    fTotalBytes += GetNormsReserved()*sizeof(int);   // fNormResult
    fTotalBytes += GetNormsReserved()*sizeof(short); // fNormParameter

    LogInfo << "Approximate Memory Size: " << fTotalBytes/1E+9
           << " GB" << std::endl;

    try {
        // Get the CPU/GPU memory for the normalization index tables.  These
        // are copied once during initialization so do not pin the CPU memory
        // into the page set.
        fNormResult.reset(new hemi::Array<int>(GetNormsReserved(),false));
        LogThrowIf(not fNormResult, "Bad NormResult Alloc");
        fNormParameter.reset(new hemi::Array<short>(GetNormsReserved(),false));
        LogThrowIf(not fNormParameter, "Bad NormParameter Alloc");
    }
    catch (...) {
        LogError << "Uncaught exception, so stopping" << std::endl;
        LogThrow("WeightNormalization -- Uncaught exception");
    }

    Reset();
}

// The destructor
Cache::Weight::Normalization::~Normalization() {}

// Reserve space for another normalization parameter.
int Cache::Weight::Normalization::ReserveNorm(int resIndex, int parIndex) {
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
        LogError << "Invalid parameter index"
               << std::endl;
        LogThrow("Parameter index out of bounds");
    }
    int newIndex = fNormsUsed++;
    if (fNormsUsed > fNormsReserved) {
        LogError << "Not enough space reserved for Norms "
                 << " Reserved: " << fNormsReserved
                 << " Requested: " << fNormsUsed
                 << std::endl;
        LogThrow("Not enough space reserved for results");
    }
    fNormResult->hostPtr()[newIndex] = resIndex;
    fNormParameter->hostPtr()[newIndex] = parIndex;
    return newIndex;
}

#include "CacheAtomicMult.h"

namespace {
    // A function to be used as the kernen on a CPU or GPU.  This must be
    // valid CUDA.  This accumulates the normalization parameters into the
    // results.
    HEMI_KERNEL_FUNCTION(HEMINormsKernel,
                         double* results,
                         const double* params,
                         const int* rIndex,
                         const short* pIndex,
                         const int NP) {
        for (int i : hemi::grid_stride_range(0,NP)) {
            const double x =  params[pIndex[i]];

            CacheAtomicMult(&results[rIndex[i]], x);
#ifndef HEMI_DEV_CODE
#ifdef CACHE_DEBUG
            if (rIndex[i] < PRINT_STEP) {
                std::cout << "Norms kernel " << i
                       << " iEvt " << rIndex[i]
                       << " iPar " << pIndex[i]
                       << " = " << x
                       << std::endl;
            }
#endif
#endif
        }
    }
}

void Cache::Weight::Normalization::Reset() {
    // Use the parent reset.
    Cache::Weight::Base::Reset();
    // Reset this class
    fNormsUsed = 0;
}

bool Cache::Weight::Normalization::Apply() {
    if (GetNormsUsed() < 1) return false;

    HEMINormsKernel normsKernel;
    hemi::launch(normsKernel,
                 fWeights.writeOnlyPtr(),
                 fParameters.readOnlyPtr(),
                 fNormResult->readOnlyPtr(),
                 fNormParameter->readOnlyPtr(),
                 GetNormsUsed());

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
// End:
