#ifndef CacheWeightGraph_hxx_seen
#define CacheWeightGraph_hxx_seen

#include "CacheWeights.h"
#include "WeightBase.h"

#include "hemi/array.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Cache {
    namespace Weight {
        class Graph;
    }
}

/// A class to apply a graph weight parameter to the cached event weights.
/// This will be used in Cache::Weights to run this type of reweighting on
/// either the host or GPU.  This graph is controlled by the value at knots
/// with linear interpolation between knots.
class Cache::Weight::Graph:
    public Cache::Weight::Base {
private:
    Cache::Parameters::Clamps& fLowerClamp;
    Cache::Parameters::Clamps& fUpperClamp;

    ///////////////////////////////////////////////////////////////////////
    /// An array of indices into the results that go for each graph.
    /// This is copied from the host to the GPU once, and is then constant.
    std::size_t fGraphsReserved;
    std::size_t fGraphsUsed;
    std::unique_ptr<hemi::Array<int>> fGraphResult;

    /// An array of indices into the parameters that go for each graph.  This
    /// is copied from the host to the GPU once, and is then constant.
    std::unique_ptr<hemi::Array<short>> fGraphParameter;

    /// An array of indices for the first knot of each graph.  This is copied
    /// from the host to the GPU once, and is then constant.
    std::unique_ptr<hemi::Array<int>> fGraphIndex;

    /// An array of the space to calculate the graphs.  This is copied from
    /// the host to the GPU once, and is then constant.
    std::size_t    fGraphSpaceReserved;
    std::size_t    fGraphSpaceUsed;
    std::unique_ptr<hemi::Array<WEIGHT_BUFFER_FLOAT>> fGraphSpace;

public:
    // Construct the class.  This should allocate all the memory on the host
    // and on the GPU.  The "results" are the total number of results to be
    // calculated (one result per event, often >1E+6).  The "parameters" are
    // the number of input parameters that are used (often ~1000).  The graphs
    // are the total number of graphs used to calculate the results (typically
    // a few per event).  The space is the total space used by all of the
    // graphs.
    Graph(Cache::Weights::Results& results,
          Cache::Parameters::Values& parameters,
          Cache::Parameters::Clamps& lowerClamps,
          Cache::Parameters::Clamps& upperClamps,
          std::size_t graphs,
          std::size_t space);

    // Deconstruct the class.  This should deallocate all the memory
    // everyplace.
    virtual ~Graph();

    /// Reinitialize the cache.  This puts it into a state to be refilled, but
    /// does not deallocate any memory.
    virtual void Reset() override;

    // Apply the kernel to the event weights.
    virtual bool Apply() override;

    /// Return the number of reserved graphs.
    std::size_t GetGraphsReserved() {return fGraphsReserved;}

    /// Return the number of graphs that were filled.
    std::size_t GetGraphsUsed() {return fGraphsUsed;}

    /// Return the number of elements reserved to hold space.
    std::size_t GetGraphSpaceReserved() const {return fGraphSpaceReserved;}

    /// Return the number of elements currently used to hold space.
    std::size_t GetGraphSpaceUsed() const {return fGraphSpaceUsed;}

    /// Add athe data for the graph.
    void AddGraph(int resultIndex, int parIndex,
                  const std::vector<double>& graphData);

    // Get the index of the parameter for the graph at sIndex.
    int GetGraphParameterIndex(int sIndex);

    // Get the parameter value for the graph at sIndex.
    double GetGraphParameter(int sIndex);

    // Get the lower (upper) bound for the graph at sIndex.
    double GetGraphLowerBound(int sIndex);
    double GetGraphUpperBound(int sIndex);

    // Get the lower (upper) clamp for the graph at sIndex.
    double GetGraphLowerClamp(int sIndex);
    double GetGraphUpperClamp(int sIndex);

    // Get the number of space in the graph at sIndex.
    int GetGraphKnotCount(int sIndex);

    // Get the function value for a knot in the graph at sIndex
    double GetGraphKnotPlace(int sIndex,int knot);
    double GetGraphKnotValue(int sIndex,int knot);

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
