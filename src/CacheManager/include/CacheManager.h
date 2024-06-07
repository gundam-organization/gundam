#ifndef CacheManager_h_seen
#define CacheManager_h_seen

#include "CacheParameters.h"

#include "CacheWeights.h"
#include "WeightNormalization.h"
#include "WeightCompactSpline.h"
#include "WeightMonotonicSpline.h"
#include "WeightUniformSpline.h"
#include "WeightGeneralSpline.h"
#include "WeightGraph.h"

#ifdef CACHE_MANAGER_USE_INDEXED_SUMS
// An older implementation of the histogram summing that may be faster for
// some (peculiar) data sets.
#include "CacheIndexedSums.h"
namespace Cache {
    using HistogramSum = Cache::IndexedSums;
}
#else
// A GPU optimized implementation of histogram summing that will be faster
// for most data sets.
#include "CacheRecursiveSums.h"
namespace Cache {
    using HistogramSum = Cache::RecursiveSums;
}
#endif

#include "SampleSet.h"
#include "EventDialCache.h"

#include "hemi/array.h"

#include <map>

namespace Cache {
    class Manager;
}

class Parameter;

/// Manage the cache calculations on the GPU.  This will work even when there
/// isn't a GPU, but it's really slow on the CPU.  This is a singleton.
class Cache::Manager {
public:
    // Get the pointer to the cache manager.  This will be a nullptr if the
    // cache is not being used.
    static Manager* Get() {return fSingleton;}

    /// Fill the cache for the current iteration.  This needs to be called
    /// before the cached weights can be used.  This is used in Propagator.cpp.
    static bool Fill();

    /// Build the cache and load it into the device.  This is used in
    /// Propagator.cpp to fill the constants needed to for the calculations.
    static bool Build(SampleSet& sampleList, EventDialCache& eventDials);

    /// Update the cache with the event and spline information.  This is
    /// called as part of Build, and can be called in other code if the cache
    /// needs to be changed.  It forages all of the information from the
    /// original sample list and event dials.
    static bool Update(SampleSet& sampleList, EventDialCache& eventDials);

    /// Flag that the Cache::Manager internal caches must be updated from the
    /// SampleSet and EventDialCache before it can be used.
    static void UpdateRequired();

    /// This returns the index of the parameter in the cache.  If the
    /// parameter isn't defined, this will return a negative value.
    static int ParameterIndex(const Parameter* fp);

    /// Return true if CUDA was used during compilation.  Necessary for
    /// running a GPU.
    static bool HasCUDA();

    /// Return true if a GPU is available at runtime.  Must have also been
    // compiled using CUDA
    static bool HasGPU(bool dump = false);

    /// Return the approximate allocated memory (e.g. on the GPU).
    std::size_t GetResidentMemory() const {return fTotalBytes;}

private:
    // This is a singleton, so the constructor is private.
    Manager(int results, int parameters,
            int norms,
            int compactSplines, int compactPoints,
            int monotonicSplines, int monotonicPoints,
            int uniformSplines, int uniformPoints,
            int generalSplines, int generalPoints,
            int graphs, int graphPoints,
            int histBins, std::string spaceType);
    static Manager* fSingleton;  // You get one guess...
    static bool fUpdateRequired; // Set to true when the cache needs an update.

    // A map between the fit parameter pointers and the parameter index used
    // by the fitter.
    static std::map<const Parameter*, int> ParameterMap;

    /// Declare all of the actual GPU caches here.  There is one GPU, so this
    /// is the ONE place that everything is collected together.

    /// The cache for parameter weights (on the GPU).
    std::unique_ptr<Cache::Parameters> fParameterCache;

    /// The cache for event weights.
    std::unique_ptr<Cache::Weights> fWeightsCache;

    /// The cache for the normalizations
    std::unique_ptr<Cache::Weight::Normalization> fNormalizations;

    /// The cache for the compact (Catmull-Rom) splines
    std::unique_ptr<Cache::Weight::CompactSpline> fCompactSplines;

    /// The cache for the monotonic splines
    std::unique_ptr<Cache::Weight::MonotonicSpline> fMonotonicSplines;

    /// The cache for the uniform splines
    std::unique_ptr<Cache::Weight::UniformSpline> fUniformSplines;

    /// The cache for the general splines
    std::unique_ptr<Cache::Weight::GeneralSpline> fGeneralSplines;

    /// The cache for the general splines
    std::unique_ptr<Cache::Weight::Graph> fGraphs;

    /// The cache for the summed histgram weights
    std::unique_ptr<Cache::HistogramSum> fHistogramsCache;

    // The rough size of all the caches.
    std::size_t fTotalBytes;

public:
    virtual ~Manager() = default;

    // Provide "internal" references to the GPU cache.  This is used in the
    // implementation, and should be ignored by most people.
    Cache::Parameters& GetParameterCache() {return *fParameterCache;}
    Cache::Weights&    GetWeightsCache() {return *fWeightsCache;}
    Cache::HistogramSum& GetHistogramsCache() {return *fHistogramsCache;}
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
