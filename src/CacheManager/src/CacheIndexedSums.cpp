#include "CacheIndexedSums.h"
#include "CacheWeights.h"

#include <iostream>
#include <exception>
#include <cmath>
#include <memory>
#include <limits>

#include <hemi/hemi_error.h>
#include <hemi/launch.h>
#include <hemi/grid_stride_range.h>

#include "Logger.h"

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[Cache::IndexedSums]"); });
#endif

// The constructor
Cache::IndexedSums::IndexedSums(Cache::Weights::Results& inputs,
                                std::size_t bins)
    : fEventWeights(inputs),
      fLowerClamp(-std::numeric_limits<double>::infinity()),
      fUpperClamp(std::numeric_limits<double>::infinity()) {
  LogExitIf((inputs.size()<1), "No bins to sum");
  LogExitIf((bins<1), "No bins to sum");

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
    LogExitIf(not fSums, "Bad Sums Alloc");
    fSums2 = std::make_unique<hemi::Array<double>>(bins,true);
    LogExitIf(not fSums2, "Bad Sums2 Alloc");
    fIndexes = std::make_unique<hemi::Array<short>>(fEventWeights.size(),false);
    LogExitIf(not fIndexes, "Bad IndexesAlloc");

  }
  catch (...) {
    LogError << "Uncaught exception, so stopping" << std::endl;
    LogExit("Uncaught exception -- not enough memory available");
  }

  // Place the cache into a default state.
  Reset();

  // Initialize the caches.  Don't try to zero everything since the
  // caches can be huge.
  std::fill(fSums->hostPtr(),
            fSums->hostPtr() + fSums->size(),
            0.0);
  std::fill(fSums2->hostPtr(),
            fSums2->hostPtr() + fSums2->size(),
            0.0);
}

// The destructor
Cache::IndexedSums::~IndexedSums() = default;

/// Reset the index sum cache to it's state immediately after construction.
void Cache::IndexedSums::Reset() {
  // Very little to do here since the indexed sum cache is zeroed with it is
  // filled.  Mark it as invalid out of an abundance of caution!
  Invalidate();
}

void Cache::IndexedSums::SetEventIndex(int event, int bin) {
  LogExitIf((event < 0), "Event index out of range");
  LogExitIf((fEventWeights.size() <= event), "Event index out of range");
  LogExitIf((bin<0), "Bin is out of range");
  LogExitIf((fSums->size() <= bin), "Bin is out of range");
  fIndexes->hostPtr()[event] = bin;
}

void Cache::IndexedSums::SetMaximumEventWeight(double maximum) {
  fUpperClamp = maximum;
}

void Cache::IndexedSums::SetMinimumEventWeight(double minimum) {
  fLowerClamp = minimum;
}

double Cache::IndexedSums::GetSum(int i) {
  LogExitIf(i<0, "Sum index out of range");
  LogExitIf((fSums->size() <= i), "Sum index out of range");
  // This odd ordering is to make sure the thread-safe hostPtr update
  // finishes before the sum is set to be valid.  The use of isnan is to
  // make sure that the optimizer doesn't reorder the statements.
  double value = fSums->hostPtr()[i];
  if (not fSumsApplied) fSumsValid = false;
  else if (not std::isnan(value)) fSumsValid = true;
  else LogExit("Cache::IndexedSums sum is nan");
  return value;
}

double Cache::IndexedSums::GetSum2(int i) {
  LogExitIf((i<0), "Sum2 index out of range");
  LogExitIf((fSums2->size()<= i), "Sum2 index out of range");
  // This odd ordering is to make sure the thread-safe hostPtr update
  // finishes before the sum is set to be valid.  The use of isfinite is to
  // make sure that the optimizer doesn't reorder the statements.
  double value = fSums2->hostPtr()[i];
  if (not fSumsApplied) fSumsValid = false;
  else if (not std::isnan(value)) fSumsValid = true;
  else LogExit("Cache::IndexedSums sum2 is nan");
  return value;
}

const double* Cache::IndexedSums::GetSumsPointer() {
  return fSums->hostPtr();
}

const double* Cache::IndexedSums::GetSums2Pointer() {
  return fSums2->hostPtr();
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
                       double* sums2,
                       const double* inputs,
                       const short* indexes,
                       const int NP) {
    for (int i : hemi::grid_stride_range(0,NP)) {
      const double v = inputs[i];
      CacheAtomicAdd(&sums[indexes[i]],v);
      CacheAtomicAdd(&sums2[indexes[i]],v*v);
    }
  }

  // A function to do the sums
  HEMI_KERNEL_FUNCTION(HEMIClampedIndexedSumKernel,
                       double* sums,
                       double* sums2,
                       const double* inputs,
                       const short* indexes,
                       const double lowerClamp,
                       const double upperClamp,
                       const int NP) {
    for (int i : hemi::grid_stride_range(0,NP)) {
      double v = inputs[i];
      if (v < lowerClamp) v = lowerClamp;
      if (v > upperClamp) v = upperClamp;
      CacheAtomicAdd(&sums[indexes[i]],v);
      CacheAtomicAdd(&sums2[indexes[i]],v*v);
    }
  }

}

bool Cache::IndexedSums::Apply() {
  // Mark the results has having changed.
  Invalidate();

  HEMIResetKernel resetKernel;
  hemi::launch(resetKernel,
               fSums->writeOnlyPtr(),
               0.0,
               fSums->size());

  hemi::launch(resetKernel,
               fSums2->writeOnlyPtr(),
               0.0,
               fSums2->size());

  if (std::isfinite(fLowerClamp) || std::isfinite(fUpperClamp)) {
    HEMIClampedIndexedSumKernel clampedIndexedSumKernel;
    hemi::launch(clampedIndexedSumKernel,
                 fSums->writeOnlyPtr(),
                 fSums2->writeOnlyPtr(),
                 fEventWeights.readOnlyPtr(),
                 fIndexes->readOnlyPtr(),
                 fLowerClamp, fUpperClamp,
                 fEventWeights.size());
  }
  else {
    HEMIIndexedSumKernel indexedSumKernel;
    hemi::launch(indexedSumKernel,
                 fSums->writeOnlyPtr(),
                 fSums2->writeOnlyPtr(),
                 fEventWeights.readOnlyPtr(),
                 fIndexes->readOnlyPtr(),
                 fEventWeights.size());
  }

  fSumsApplied = true;

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
// End:
