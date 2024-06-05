#ifndef CacheRecursiveSums_h_seen
#define CacheRecursiveSums_h_seen

#include "CacheWeights.h"

#include "hemi/array.h"

#include <cstdint>
#include <memory>

namespace Cache {
    class RecursiveSums;
}

/// A class to accumulate the sum of event weights into the histogram bins on
/// the GPU (or CPU).  This provides a parallel reduction (a sum) without
/// using atomic operations by interleaving pairwise sums and "recursively"
/// calling the kernel.  On a GPU, it results in about a x10 speed up over a
/// naive sum using atomic operations, but requires more GPU global memory
/// since it uses an internal work space.
///
/// The accumulation runs the "sum" kernel O(log N) times where N is the
/// number of entries in the maximum histogram bin.  The sum is "logically"
/// recursive, but the code actually iterates by calling the summing kernel
/// multiple times.
class Cache::RecursiveSums {
private:
    // Save the event weight cache reference for later use.  This is provided
    // to the constructor.
    Cache::Weights::Results& fEventWeights;

    // The histogram bin index for each entry in the fWeights array (this is
    // the same size as fEventWeights.  This is filled before initialization.
    std::unique_ptr<hemi::Array<short>> fIndexes;

    // An internal buffer holding the offset of each histogram bin in the
    // event index array.  The start of the events in bin "N" will be
    // fBinOffsets[N], and the end will be fBinOffsets[N+1], so looping over
    // the events will be "for(i=fBinOffsets[N]; i<fBinOffsets[N+1]; ++i)"
    std::unique_ptr<hemi::Array<int>> fBinOffsets;

    // An internal buffer storing the index of each entry associated in the
    // fWeights array associated with the histogram bin.  The fBinOffsets
    // field defines which bin the entries are associated with.
    std::unique_ptr<hemi::Array<int>> fEventIndexes;

    // An internal buffer used to do the recursive sum.  This starts as a copy
    // of the fEventWeights input array (reordered by bin index), and is
    // mutated until the sums can be read.
    std::unique_ptr<hemi::Array<double>> fWorkBuffer;

    // An internal buffer used to map the fWorkBuffer index to the histogram
    // bin index.
    std::unique_ptr<hemi::Array<short>> fBinIndexes;

    // The maximum number of entries in any bin. This will determine the number
    // of iterations needed to calculate the sum.
    int fMaxEntries;

    // The accumulated weights for each histogram bin.
    std::unique_ptr<hemi::Array<double>> fSums;

    // The accumulated weights for each histogram bin.
    std::unique_ptr<hemi::Array<double>> fSums2;

    // The lower bound for any individual entry in the fWeights array.  This
    // is a global event weight clamp.
    double fLowerClamp;

    // The upper bound for any individual entry in the fWeights array.  This
    // is a global event weight clamp.
    double fUpperClamp;

    // Cache of whether the result values in memory are valid.
    bool fSumsValid;

    /// The (approximate) amount of memory required on the GPU.
    std::size_t fTotalBytes{};

public:
    RecursiveSums(Cache::Weights::Results& eventWeight,
                  std::size_t bins);

    /// Deconstruct the class.  This should deallocate all the memory
    /// everyplace.
    virtual ~RecursiveSums();

    /// Reinitialize the cache.  This puts it into a state to be refilled, but
    /// does not deallocate any memory.
    void Reset();

    /// Initialize the internal buffers for the cache for all of the events.
    /// This builds all the maps between histogram bin and event index (and
    /// counts the number of entries in each bin.
    void Initialize();

    /// Return the approximate allocated memory (e.g. on the GPU).
    std::size_t GetResidentMemory() const {return fTotalBytes;}

    // Assigns the bin number that an event will be added to.
    void SetEventIndex(int event, int bin);

    // Set the maximum event weight to be applied as an upper clamp during the
    // sum.  (Default: infinity).
    void SetMaximumEventWeight(double maximum);

    // Set the minimum event weight to be applied as an upper clamp during the
    // sum.  (Default: negative infinity).
    void SetMinimumEventWeight(double minimum);

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
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
#endif
