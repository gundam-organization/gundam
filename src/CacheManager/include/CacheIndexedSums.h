#ifndef CacheIndexedSums_h_seen
#define CacheIndexedSums_h_seen

#include "CacheWeights.h"

#include "hemi/array.h"

#include <cstdint>
#include <memory>

namespace Cache {
    class IndexedSums;
}

/// A class to calculate and cache a bunch of events weights.  This is where
/// the actual calculation is controled for the GPU.  It's used by the cache
/// manager when the GPU needs to be fired up.
class Cache::IndexedSums {
private:
    // Save the event weight cache reference for later use
    Cache::Weights::Results& fEventWeights;

    // The histogram bin index for each entry in the fWeights array (this is
    // the same size as fEventWeights.
    std::unique_ptr<hemi::Array<short>> fIndexes;

    // The accumulated weights for each histogram bin.
    std::unique_ptr<hemi::Array<double>> fSums;

    // The accumulated weights for each histogram bin.
    std::unique_ptr<hemi::Array<double>> fSums2;

    // Cache of whether the result values in memory are valid.
    bool fSumsValid;

    /// The (approximate) amount of memory required on the GPU.
    std::size_t fTotalBytes{};

public:
    IndexedSums(Cache::Weights::Results& eventWeight,
               std::size_t bins);

    /// Deconstruct the class.  This should deallocate all the memory
    /// everyplace.
    virtual ~IndexedSums();

    /// Reinitialize the cache.  This puts it into a state to be refilled, but
    /// does not deallocate any memory.
    void Reset();

    /// Return the approximate allocated memory (e.g. on the GPU).
    std::size_t GetResidentMemory() const {return fTotalBytes;}

    // Assigns the bin number that an event will be added to.
    void SetEventIndex(int event, int bin);

    /// Return the number of histogram bins that are accumulated.
    std::size_t GetSumCount() const {return fSums->size();}

    /// Calculate the results and save them for later use.  This copies the
    /// results from the GPU to the CPU.
    virtual bool Apply();

    /// Get the sum for index i from host memory.  This might trigger a copy
    /// from the device if that is necessary.
    double GetSum(int i);

    /// Get the sum squared for index i from host memory.  This might trigger
    /// a copy from the device if that is necessary.
    double GetSum2(int i);

    /// The pointer to the array of sums on the host.
    const double* GetSumsPointer();

    /// The pointer to the array of sums squared on the host.
    const double* GetSums2Pointer();

    /// A pointer to the validity flag.
    bool* GetSumsValidPointer();

};

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
#endif
