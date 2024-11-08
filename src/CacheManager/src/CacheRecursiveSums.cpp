#include "CacheRecursiveSums.h"
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
LoggerInit([]{ Logger::setUserHeaderStr("[Cache::RecursiveSums]"); });
#endif

// The constructor
Cache::RecursiveSums::RecursiveSums(Cache::Weights::Results& inputs,
                                    std::size_t bins)
    : fEventWeights(inputs),
      fLowerClamp(-std::numeric_limits<double>::infinity()),
      fUpperClamp(std::numeric_limits<double>::infinity()) {
  LogExitIf((inputs.size()<1), "No bins to sum");
  LogExitIf((bins<1), "No bins to sum");

  LogInfo << "Cached RecursiveSums -- bins reserved: "
          << bins
          << std::endl;
  fTotalBytes += bins*sizeof(double);                   // fSums
  fTotalBytes += bins*sizeof(double);                   // fSums2
  fTotalBytes += fEventWeights.size()*sizeof(short);    // fIndexes;
  fTotalBytes += (bins+1)*sizeof(int);                  // fBinOffsets
  fTotalBytes += fEventWeights.size()*sizeof(int);      // fEventIndexes
  fTotalBytes += fEventWeights.size()*sizeof(double);   // fWorkBuffer
  fTotalBytes += fEventWeights.size()*sizeof(short);    // fBinIndexes

  LogInfo << "Cached RecursiveSums -- approximate memory size: "
          << double(fTotalBytes)/1E+6
          << " MB" << std::endl;

  try {
    // Get CPU/GPU memory for the results and their initial values.  The
    // results are copied every time, so pin the CPU memory into the page
    // set.  The initial values are seldom changed, so they are not
    // pinned.
    fSums = std::make_unique<hemi::Array<double>>(bins,true);
    LogExitIf(not fSums, "Bad Sums Alloc");
    fSums2 = std::make_unique<hemi::Array<double>>(bins,true);
    LogExitIf(not fSums2, "Bad Sums2 Alloc");

    fIndexes = std::make_unique<hemi::Array<short>>(fEventWeights.size(),false);
    LogExitIf(not fIndexes, "Bad Indexes Alloc");
    fBinOffsets = std::make_unique<hemi::Array<int>>(bins+1,false);
    LogExitIf(not fBinOffsets, "Bad BinOffsets Alloc");
    fEventIndexes = std::make_unique<hemi::Array<int>>(fEventWeights.size(),false);
    LogExitIf(not fEventIndexes,"Bad EventIndexes Alloc");
    fWorkBuffer = std::make_unique<hemi::Array<double>>(fEventWeights.size(),false);
    LogExitIf(not fWorkBuffer,"Bad WorkBuffer Alloc");
    fBinIndexes = std::make_unique<hemi::Array<short>>(fEventWeights.size(),false);
    LogExitIf(not fBinIndexes,"Bad BinIndexes Alloc");
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
Cache::RecursiveSums::~RecursiveSums() = default;

/// Reset the index sum cache to it's state immediately after construction.
void Cache::RecursiveSums::Reset() {
  // Very little to do here since the indexed sum cache is zeroed with it is
  // filled.  Mark it as invalid out of an abundance of caution!
  Invalidate();
}

/// Build the internal tables after all the events are filled.  This can be
/// slow, but only happens once, and isn't slow on the time scale of start-up.
void Cache::RecursiveSums::Initialize() {

  // Zero the number of entries in each bin.  This makes sure that all
  // bins are represented, even if they don't have events.
  struct BinEntryCount{
    int binIdx{-1};
    int entryCount{0};
  };
  std::vector<BinEntryCount> binEntryCountList;

  binEntryCountList.reserve( fSums->size() );
  for (int b = 0; b < fSums->size(); ++b){
    binEntryCountList.emplace_back();
    binEntryCountList.back().binIdx = b;
  }

  // Count the number of entries in each bin
  short* idx = fIndexes->hostPtr();
  for( int e = 0; e < fIndexes->size(); ++e ){
    binEntryCountList.at(idx[e]).entryCount += 1;
  }


  // Find the maximum number of entries in any bin.
  fMaxEntries = 0;
  for ( auto& count : binEntryCountList ) {
    fMaxEntries = std::max(fMaxEntries, count.entryCount);
  }

  // Fill the offsets for each histogram bin.  This also makes sure that all
  // the bins exist.  There will be a problem in one of the bins is empty.
  {
    int bin = 0;
    int offset = 0;
    for ( auto& count : binEntryCountList) {
      LogExitIf(bin != count.binIdx, "Bin number mismatch: bin=" << bin << " / count->first=" << count.binIdx);
      fBinOffsets->hostPtr()[bin] = offset;
      offset += count.entryCount;
      ++bin;
    }
    fBinOffsets->hostPtr()[bin] = offset;
  }

  // Build a simple lookup associating each entry in the fWorkBuffer with
  // a histogram bin.
  for (int bin = 0; bin < fBinOffsets->size()-1; ++bin) {
    for (int i = fBinOffsets->hostPtr()[bin];
         i < fBinOffsets->hostPtr()[bin+1]; ++i) {
      fBinIndexes->hostPtr()[i] = bin;
    }
  }

  // Build the map between the work buffer entry and the weight entry.
  for (int b = 0; b < fSums->size(); ++b) binEntryCountList[b].entryCount = 0;
  for (int entry = 0; entry < fIndexes->size(); ++entry) {
    int bin = fIndexes->hostPtr()[entry];
    int offset = fBinOffsets->hostPtr()[bin] + binEntryCountList[bin].entryCount;
    fEventIndexes->hostPtr()[offset] = entry;
    ++binEntryCountList[bin].entryCount;
  }

}

void Cache::RecursiveSums::SetEventIndex(int event, int bin) {
  LogExitIf((event < 0), "Event index out of range");
  LogExitIf((fEventWeights.size() <= event), "Event index out of range");
  LogExitIf((bin<0), "Bin is out of range");
  LogExitIf((fSums->size() <= bin), "Bin is out of range");
  fIndexes->hostPtr()[event] = bin;
}

void Cache::RecursiveSums::SetMaximumEventWeight(double maximum) {
  fUpperClamp = maximum;
}

void Cache::RecursiveSums::SetMinimumEventWeight(double minimum) {
  fLowerClamp = minimum;
}

double Cache::RecursiveSums::GetSum(int i) {
  LogExitIf(i<0, "Sum index out of range");
  LogExitIf((fSums->size() <= i), "Sum index out of range");
  // This odd ordering is to make sure the thread-safe hostPtr update
  // finishes before the sum is set to be valid.  The use of isnan is to
  // make sure that the optimizer doesn't reorder the statements.
  double value = fSums->hostPtr()[i];
  if (not fSumsApplied) fSumsValid = false;
  else if (not std::isnan(value)) fSumsValid = true;
  else LogExit("Cache::RecursiveSums sum is nan");
  return value;
}

double Cache::RecursiveSums::GetSum2(int i) {
  LogExitIf((i<0), "Sum2 index out of range");
  LogExitIf((fSums2->size()<= i), "Sum2 index out of range");
  // This odd ordering is to make sure the thread-safe hostPtr update
  // finishes before the sum is set to be valid.  The use of isfinite is to
  // make sure that the optimizer doesn't reorder the statements.
  double value = fSums2->hostPtr()[i];
  if (not fSumsApplied) fSumsValid = false;
  else if (not std::isnan(value)) fSumsValid = true;
  else LogExit("Cache::RecursiveSums sum2 is nan");
  return value;
}

const double* Cache::RecursiveSums::GetSumsPointer() {
  return fSums->hostPtr();
}

const double* Cache::RecursiveSums::GetSums2Pointer() {
  return fSums2->hostPtr();
}

bool* Cache::RecursiveSums::GetSumsValidPointer() {
  return &fSumsValid;
}

// Define CACHE_DEBUG to get lots of output from the host
#undef CACHE_DEBUG

#include "CacheAtomicAdd.h"

namespace {
  // A function to copy the event weights into the internal work buffer
  HEMI_KERNEL_FUNCTION(HEMIFillWorkBuffer,
                       double* buffer,
                       const double* inputs,
                       const int* indexes,
                       const double lowerClamp,
                       const double upperClamp,
                       const int NP) {
    for (int i : hemi::grid_stride_range(0,NP)) {
      double weight = inputs[indexes[i]];
      if (weight < lowerClamp) weight = lowerClamp;
      if (weight > upperClamp) weight = upperClamp;
      buffer[i] = weight;
    }
  }

  // A function to copy the squared event weights into the internal work
  // buffer
  HEMI_KERNEL_FUNCTION(HEMIFillWorkBuffer2,
                       double* buffer,
                       const double* inputs,
                       const int* indexes,
                       const double lowerClamp,
                       const double upperClamp,
                       const int NP) {
    for (int i : hemi::grid_stride_range(0,NP)) {
      double weight = inputs[indexes[i]];
      if (weight < lowerClamp) weight = lowerClamp;
      if (weight > upperClamp) weight = upperClamp;
      buffer[i] = weight*weight;
    }
  }

  // Add pairs of values together.  This doesn't need a lock!  It is called
  // with ever decreasing strides from twice the number of entries in the
  // maximum bin down to one.  The stride is always a power of two.  During
  // the first call, the threads will be close to fully occupied, but
  // eventually the thread efficiency drops (by half) at each iteration.
  HEMI_KERNEL_FUNCTION(HEMIStridingSum,
                       double* buffer,
                       const short* binIndexes,
                       const int* binOffsets,
                       int NP,
                       int stride) {
    for (int i : hemi::grid_stride_range(0,NP)) {
      const int bin = binIndexes[i];
      const int binOffset = binOffsets[bin];
      const int binMax = binOffsets[bin+1];
      const int binAdd = i+stride;
      if (binAdd < binMax && binAdd < binOffset+2*stride) {
        buffer[i] += buffer[binAdd];
      }
    }
  }

  // A function to copy the final sums into the output.
  HEMI_KERNEL_FUNCTION(HEMICopyResults,
                       double* sums,
                       const double* buffer,
                       const int* offsets,
                       int NB) {
    for (int i : hemi::grid_stride_range(0, NB)) {
      sums[i] = buffer[offsets[i]];
    }
  }
}

bool Cache::RecursiveSums::Apply() {
  // Mark the results has having changed.
  Invalidate();

  ///////////////////////////////////////////////////
  // Calculate the sum
  ///////////////////////////////////////////////////
  HEMIFillWorkBuffer fillWorkBuffer;
  hemi::launch(fillWorkBuffer,
               fWorkBuffer->writeOnlyPtr(),
               fEventWeights.readOnlyPtr(),
               fEventIndexes->readOnlyPtr(),
               fLowerClamp, fUpperClamp,
               fEventWeights.size());

  // Find the maximum stride that will be needed to handle the bin with the
  // most entries.
  int stride = 1;
  while (2*stride < fMaxEntries) stride *= 2;

  // Sum pairs of weights until there is only one.  There can be only one.
  HEMIStridingSum stridingSum;
  while (stride > 0) {
    hemi::launch(stridingSum,
                 fWorkBuffer->writeOnlyPtr(),
                 fBinIndexes->readOnlyPtr(),
                 fBinOffsets->readOnlyPtr(),
                 fWorkBuffer->size(),
                 stride);
    stride /= 2;
  }

  // Copy the sums into the output.
  HEMICopyResults copyResults;
  hemi::launch(copyResults,
               fSums->writeOnlyPtr(),
               fWorkBuffer->readOnlyPtr(),
               fBinOffsets->readOnlyPtr(),
               fSums->size());

  ///////////////////////////////////////////////////
  // Calculate the sum squared (almost copy and paste).
  ///////////////////////////////////////////////////
  HEMIFillWorkBuffer2 fillWorkBuffer2;
  hemi::launch(fillWorkBuffer2,
               fWorkBuffer->writeOnlyPtr(),
               fEventWeights.readOnlyPtr(),
               fEventIndexes->readOnlyPtr(),
               fLowerClamp, fUpperClamp,
               fEventWeights.size());
  stride = 1;
  while (2*stride < fMaxEntries) stride *= 2;
  while (stride > 0) {
    hemi::launch(stridingSum,
                 fWorkBuffer->writeOnlyPtr(),
                 fBinIndexes->readOnlyPtr(),
                 fBinOffsets->readOnlyPtr(),
                 fWorkBuffer->size(),
                 stride);
    stride /= 2;
  }
  hemi::launch(copyResults,
               fSums2->writeOnlyPtr(),
               fWorkBuffer->readOnlyPtr(),
               fBinOffsets->readOnlyPtr(),
               fSums2->size());

  fSumsApplied = true;

  return true;
}

// An MIT Style License

// Copyright (c) 2024 Clark McGrew

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
