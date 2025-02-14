#ifndef CacheUniformSpline_hxx_seen
#define CacheUniformSpline_hxx_seen

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
    class UniformSpline;
  }
}

/// A class apply a splined weight parameter to the cached event weights.
/// This will be used in Cache::Weights to run the GPU for this type of
/// reweighting.  This spline is controlled by the value and slope at
/// uniformly spaced knots.
class Cache::Weight::UniformSpline:
    public Cache::Weight::Base {
private:
  Cache::Parameters::Clamps& fLowerClamp;
  Cache::Parameters::Clamps& fUpperClamp;

  ///////////////////////////////////////////////////////////////////////
  /// An array of indices into the results that go for each spline.
  /// This is copied from the CPU to the GPU once, and is then constant.
  std::size_t fSplinesReserved;
  std::size_t fSplinesUsed;
  std::unique_ptr<hemi::Array<int>> fSplineResult;

  /// An array of indices into the parameters that go for each spline.  This
  /// is copied from the CPU to the GPU once, and is then constant.
  std::unique_ptr<hemi::Array<short>> fSplineParameter;

  /// An array of indices for the first knot of each spline.  This is copied
  /// from the CPU to the GPU once, and is then constant.
  std::unique_ptr<hemi::Array<int>> fSplineIndex;

  /// An array of the knots to calculate the splines.  This is copied from
  /// the CPU to the GPU once, and is then constant.
  std::size_t    fSplineSpaceReserved;
  std::size_t    fSplineSpaceUsed;
  std::unique_ptr<hemi::Array<WEIGHT_BUFFER_FLOAT>> fSplineSpace;

public:
  // A static method to return the number of knots that will be used by this
  // spline.
  static int FindPoints(const TSpline3* s);

  // Construct the class.  This should allocate all the memory on the host
  // and on the GPU.  The "results" are the total number of results to be
  // calculated (one result per event, often >1E+6).  The "parameters" are
  // the number of input parameters that are used (often ~1000).  The norms
  // are the total number of normalization parameters (typically a few per
  // event) used to calculate the results.  The splines are the total number
  // of spline parameters (with uniform knot spacing) used to calculate the
  // results (typically a few per event).  The knots are the total number of
  // knots in all of the uniform splines (e.g. For 1000 splines with 7
  // knots for each spline, knots is 7000).
  UniformSpline(Cache::Weights::Results& results,
                Cache::Parameters::Values& parameters,
                Cache::Parameters::Clamps& lowerClamps,
                Cache::Parameters::Clamps& upperClamps,
                std::size_t splines,std::size_t knots,
                std::string spaceOption);

  // Deconstruct the class.  This should deallocate all the memory
  // everyplace.
  virtual ~UniformSpline();

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

  /// Add a spline data.
  void AddSpline(int resultIndex, int parIndex, const std::vector<double>& splineData);

  // Get the index of the parameter for the spline at sIndex.
  int GetSplineParameterIndex(int sIndex);

  // Get the parameter value for the spline at sIndex.
  double GetSplineParameter(int sIndex);

  // Get the lower (upper) bound for the spline at sIndex.
  double GetSplineLowerBound(int sIndex);
  double GetSplineUpperBound(int sIndex);

  // Get the lower (upper) clamp for the spline at sIndex.
  double GetSplineLowerClamp(int sIndex);
  double GetSplineUpperClamp(int sIndex);

  // Get the number of knots in the spline at sIndex.
  int GetSplineKnotCount(int sIndex);

  // Get the function value for a knot in the spline at sIndex
  double GetSplineKnotValue(int sIndex,int knot);
  double GetSplineKnotSlope(int sIndex,int knot);

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
