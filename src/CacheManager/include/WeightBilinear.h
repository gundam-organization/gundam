#ifndef CacheBilinear_hxx_seen
#define CacheBilinear_hxx_seen

#include "CacheWeights.h"
#include "WeightBase.h"

#include "hemi/array.h"

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
  std::size_t fReserved;
  std::size_t fUsed;
  std::unique_ptr<hemi::Array<int>> fResult;

  /// An array of indices into the parameters that go for each surface.  This
  /// is copied from the CPU to the GPU once, and is then constant.
  std::unique_ptr<hemi::Array<short>> fParameter;

  /// An array of indices for the first data for each surface.  This is
  /// copied from the CPU to the GPU once, and is then constant.
  std::unique_ptr<hemi::Array<int>> fIndex;

  /// An array for the data in the surfaces.  This is copied from
  /// the CPU to the GPU once, and is then constant.
  std::size_t    fSpaceReserved;
  std::size_t    fSpaceUsed;
  std::unique_ptr<hemi::Array<WEIGHT_BUFFER_FLOAT>> fData;

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

  /// Return the number of parameters using bilinear interpolation
  /// are reserved.
  std::size_t GetReserved() {return fReserved;}

  /// Return the number of parameters using a spline with uniform knots that
  /// are used.
  std::size_t GetUsed() {return fUsed;}

  /// Return the number of elements reserved to hold knots.
  std::size_t GetSpaceReserved() const {return fSpaceReserved;}

  /// Return the number of elements currently used to hold knots.
  std::size_t GetSpaceUsed() const {return fSpaceUsed;}

  /// Add the data for the spline.
  void AddData(int resultIndex, int parIndex1, int parIndex2,
               const std::vector<double>& splineData);

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
// End:
#endif
