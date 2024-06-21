#ifndef CacheBilinear_hxx_seen
#define CacheBilinear_hxx_seen

#include "CacheWeights.h"
#include "WeightBase.h"

#include "hemi/array.h"

class SplineDial;

#include <TSpline.h>

#include <cstdint>
#include <memory>
#include <vector>


namespace Cache {
    namespace Weight {
        class Bilinear;
    }
}

/// A class to apply a 2D linear interpolation weight parameter to the cached
/// event weights.
class Cache::Weight::Bilinear:
    public Cache::Weight::Base {
private:
    Cache::Parameters::Clamps& fLowerClamp;
    Cache::Parameters::Clamps& fUpperClamp;

    ///////////////////////////////////////////////////////////////////////
    /// An array of indices into the results that go for each surface.
    /// This is copied from the CPU to the GPU once, and is then constant.
    std::size_t fSplinesReserved;
    std::size_t fSplinesUsed;
    std::unique_ptr<hemi::Array<int>> fSplineResult;

    /// An array of indices into the parameters that go for each surface.  This
    /// is copied from the CPU to the GPU once, and is then constant.
    std::unique_ptr<hemi::Array<short>> fSplineParameter;

    /// An array of indices for the first data for each surface.  This is
    /// copied from the CPU to the GPU once, and is then constant.
    std::unique_ptr<hemi::Array<int>> fSplineIndex;

    /// An array for the data in the surfaces.  This is copied from
    /// the CPU to the GPU once, and is then constant.
    std::size_t    fSplineSpaceReserved;
    std::size_t    fSplineSpaceUsed;
    std::unique_ptr<hemi::Array<WEIGHT_BUFFER_FLOAT>> fSplineData;

public:

    // Construct the class.  This should allocate all the memory on the host
    // and on the GPU.  The "results" are the total number of results to be
    // calculated (one result per event, often >1E+6).  The "parameters" are
    // the number of input parameters that are used (often ~1000).  The
    // splines are the total number of 2D surfaces (typically a couple per
    // event).  The space is the number of elements (knots and dimension
    // coordinates) that are needed to hold all of the surface data.
    Bilinear(Cache::Weights::Results& results,
             Cache::Parameters::Values& parameters,
             Cache::Parameters::Clamps& lowerClamps,
             Cache::Parameters::Clamps& upperClamps,
             std::size_t splines,
             std::size_t space);

    // Deconstruct the class.  This should deallocate all the memory
    // everyplace.
    virtual ~Bilinear();

    /// Reinitialize the cache.  This puts it into a state to be refilled, but
    /// does not deallocate any memory.
    virtual void Reset() override;

    // Apply the kernel to the event weights.
    virtual bool Apply() override;

    /// Return the number of parameters using a spline with uniform knots that
    /// are reserved.
    std::size_t GetSplinesReserved() {return fSplinesReserved;}

    /// Return the number of parameters using a spline with uniform knots that
    /// are used.
    std::size_t GetSplinesUsed() {return fSplinesUsed;}

    /// Return the number of elements reserved to hold knots.
    std::size_t GetSplineSpaceReserved() const {return fSplineSpaceReserved;}

    /// Return the number of elements currently used to hold knots.
    std::size_t GetSplineSpaceUsed() const {return fSplineSpaceUsed;}

    /// Add athe data for the spline.
    void AddSpline(int resultIndex, int parIndex1, int parIndex2,
                   const std::vector<double>& splineData);

    // Get the index of the parameter for the spline at sIndex.
    int GetSplineParameterIndex(int sIndex);

    // Get the parameter1 value for the spline at sIndex.
    double GetSplineParameter1(int sIndex);

    // Get the parameter2 value for the spline at sIndex.
    double GetSplineParameter2(int sIndex);

    // Get the lower (upper) bound for the spline at sIndex.
    double GetSplineLowerBound(int sIndex);
    double GetSplineUpperBound(int sIndex);

    // Get the lower (upper) clamp for the spline at sIndex.
    double GetSplineLowerClamp(int sIndex);
    double GetSplineUpperClamp(int sIndex);

    // Get the number of knots in the spline at sIndex.
    int GetSplineSpaceCount(int sIndex);
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
// compile-command:"$(git rev-parse --show-toplevel)/cmake/scripts/gundam-build.sh"
// End:
#endif
