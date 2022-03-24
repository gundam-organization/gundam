#ifndef CacheManager_h_seen
#define CacheManager_h_seen

#include "CacheWeights.h"
#include "WeightNormalization.h"
#include "WeightCompactSpline.h"

#include "FitSampleSet.h"

#include "hemi/array.h"

#include <map>

namespace Cache {
    class Manager;
}

class FitParameter;

class Cache::Manager {
public:
    // Get the pointer to the cache manager.  This will be a nullptr if the
    // cache is not being used.
    static Manager* Get() {return fSingleton;}

    // Fill the cache for the current iteration.
    static bool Fill();

    // Build the cache and load it into the device.
    static bool Build(FitSampleSet& sampleList);

    Cache::Parameters& GetParameterCache() {return *fParameterCache;}
    Cache::Weights& GetWeightsCache() {return *fWeightsCache;}

private:
    // This is a singleton, so the constructor is private.
    Manager(int results, int parameters,
            int norms,
            int compactSplines, int compactPoints,
            int uniformSplines, int uniformPoints,
            int generalSplines, int generalPoints);

    static Manager* fSingleton;  // You get one guess...

    // A map between the fit parameter pointers and the parameter index used
    // by the fitter.
    static std::map<const FitParameter*, int> ParameterMap;

    // Determine the type of spline cache to use.  The possible results are
    // "compactSpline", "uniformSpline", "generalSpline", or
    // "this-cannot-happen".
    static std::string SplineType(const TSpline3* s);

    /// Declare all of the actual GPU caches here.  There is one GPU, so this
    /// is the ONE place that everything is collected together.

    /// The cache for parameter weights (on the GPU).
    std::unique_ptr<Cache::Parameters> fParameterCache;

    /// The cache for event weights.
    std::unique_ptr<Cache::Weights> fWeightsCache;

    /// The cache for the normalizations
    std::unique_ptr<Cache::Weight::Normalization> fNormalizations;

    /// The cache for the compact splines
    std::unique_ptr<Cache::Weight::CompactSpline> fCompactSplines;

    /// The cache for the uniform splines (really compact splines for now).
    std::unique_ptr<Cache::Weight::CompactSpline> fUniformSplines;

    /// The cache for the general splines (really compact splines for now).
    std::unique_ptr<Cache::Weight::CompactSpline> fGeneralSplines;

    std::size_t fTotalBytes;

public:
    ~Manager();

    /// Return the approximate allocated memory (e.g. on the GPU).
    std::size_t GetResidentMemory() const {return fTotalBytes;}

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
