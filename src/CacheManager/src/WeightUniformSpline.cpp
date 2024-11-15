#include "CacheWeights.h"
#include "WeightBase.h"
#include "WeightUniformSpline.h"

#include <algorithm>
#include <iostream>
#include <exception>
#include <limits>
#include <cmath>

#include <hemi/hemi_error.h>
#include <hemi/launch.h>
#include <hemi/grid_stride_range.h>

#include "Logger.h"


// The constructor
Cache::Weight::UniformSpline::UniformSpline(
    Cache::Weights::Results& weights,
    Cache::Parameters::Values& parameters,
    Cache::Parameters::Clamps& lowerClamps,
    Cache::Parameters::Clamps& upperClamps,
    std::size_t splines, std::size_t knots,
    std::string spaceOption)
    : Cache::Weight::Base("uniformSpline",weights,parameters),
      fLowerClamp(lowerClamps), fUpperClamp(upperClamps),
      fSplinesReserved(splines), fSplinesUsed(0),
      fSplineSpaceReserved(knots), fSplineSpaceUsed(0) {

  LogInfo << "Reserved " << GetName() << " Splines: "
          << GetSplinesReserved() << std::endl;
  if (GetSplinesReserved() < 1) return;

  fTotalBytes += GetSplinesReserved()*sizeof(int);      // fSplineResult
  fTotalBytes += GetSplinesReserved()*sizeof(short);    // fSplineParameter
  fTotalBytes += (1+GetSplinesReserved())*sizeof(int);  // fSplineIndex

  // Calculate the space needed to store the spline data.  This needs
  // to know how the spline data is packed for CalculateUniformSpline.
  if (spaceOption == "points") {
    fSplineSpaceReserved = 2*fSplinesReserved + 2*fSplineSpaceReserved;
  }
  else {
    LogThrowIf(spaceOption != "space",
               "Invalid space option for compact splines");
  }

  LogInfo << "Reserved " << GetName()
          << " Spline Knots: " << GetSplineSpaceReserved()
          << std::endl;
  fTotalBytes += GetSplineSpaceReserved()*sizeof(WEIGHT_BUFFER_FLOAT);  // fSplineKnots

  LogInfo << "Approximate Memory Size for " << GetName()
          << ": " << fTotalBytes/1E+9
          << " GB" << std::endl;

  try {
    // Get the CPU/GPU memory for the spline index tables.  These are
    // copied once during initialization so do not pin the CPU memory into
    // the page set.
    fSplineResult.reset(new hemi::Array<int>(GetSplinesReserved(),false));
    LogThrowIf(not fSplineResult, "Bad SplineResult alloc");
    fSplineParameter.reset(
        new hemi::Array<short>(GetSplinesReserved(),false));
    LogThrowIf(not fSplineParameter, "Bad SplineParameter alloc");
    fSplineIndex.reset(new hemi::Array<int>(1+GetSplinesReserved(),false));
    LogThrowIf(not fSplineIndex, "Bad SplineIndex alloc");

    // Get the CPU/GPU memory for the spline knots.  This is copied once
    // during initialization so do not pin the CPU memory into the page
    // set.
    fSplineSpace.reset(
        new hemi::Array<WEIGHT_BUFFER_FLOAT>(GetSplineSpaceReserved(),false));
    LogThrowIf(not fSplineSpace, "Bad SplineSpace alloc");
  }
  catch (...) {
    LogError << "Uncaught exception, so stopping" << std::endl;
    LogThrow("WeightUniformSpline -- Uncaught exception");
  }

  // Initialize the caches.  Don't try to zero everything since the
  // caches can be huge.
  Reset();
  fSplineIndex->hostPtr()[0] = 0;
}

// The destructor
Cache::Weight::UniformSpline::~UniformSpline() {}

int Cache::Weight::UniformSpline::FindPoints(const TSpline3* s) {
  return s->GetNp();
}

void Cache::Weight::UniformSpline::AddSpline(int resIndex, int parIndex,
                                             const std::vector<double>& splineData) {
  if (resIndex < 0) {
    LogError << "Invalid result index"
             << std::endl;
    LogThrow("Negative result index");
  }
  if (fWeights.size() <= resIndex) {
    LogError << "Invalid result index"
             << std::endl;
    LogThrow("Result index out of bounds");
  }
  if (parIndex < 0) {
    LogError << "Invalid parameter index"
             << std::endl;
    LogThrow("Negative parameter index");
  }
  if (fParameters.size() <= parIndex) {
    LogError << "Invalid parameter index"
             << std::endl;
    LogThrow("Parameter index out of bounds");
  }
  int points = splineData.size();
  if (points < 8) {
    LogError << "Insufficient points in spline"
             << std::endl;
    LogThrow("Invalid number of spline points");
  }
  int newIndex = fSplinesUsed++;
  if (fSplinesUsed > fSplinesReserved) {
    LogError << "Not enough space reserved for splines"
             << std::endl;
    LogThrow("Not enough space reserved for splines");
  }
  fSplineResult->hostPtr()[newIndex] = resIndex;
  fSplineParameter->hostPtr()[newIndex] = parIndex;
  if (fSplineIndex->hostPtr()[newIndex] != fSplineSpaceUsed) {
    LogError << "Last spline knot index should be at old end of splines"
             << std::endl;
    LogThrow("Problem with control indices");
  }
  int knotIndex = fSplineSpaceUsed;
  fSplineSpaceUsed += points;
  if (fSplineSpaceUsed > fSplineSpaceReserved) {
    LogError << "Not enough space reserved for spline knots"
             << " -> SplineSpaceReserved = " << fSplineSpaceReserved
             << " / SplineSpaceUsed = " << fSplineSpaceUsed
             << std::endl;
    LogThrow("Not enough space reserved for spline knots");
  }
  fSplineIndex->hostPtr()[newIndex+1] = fSplineSpaceUsed;
  for (std::size_t i = 0; i<splineData.size(); ++i) {
    fSplineSpace->hostPtr()[knotIndex+i] = splineData.at(i);
  }

}

int Cache::Weight::UniformSpline::GetSplineParameterIndex(int sIndex) {
  if (sIndex < 0) {
    LogThrow("Spline index invalid");
  }
  if (GetSplinesUsed() <= sIndex) {
    LogThrow("Spline index invalid");
  }
  return fSplineParameter->hostPtr()[sIndex];
}

double Cache::Weight::UniformSpline::GetSplineParameter(int sIndex) {
  int i = GetSplineParameterIndex(sIndex);
  if (i<0) {
    LogThrow("Spline parameter index out of bounds");
  }
  if (fParameters.size() <= i) {
    LogThrow("Spline parameter index out of bounds");
  }
  return fParameters.hostPtr()[i];
}

int Cache::Weight::UniformSpline::GetSplineKnotCount(int sIndex) {
  if (sIndex < 0) {
    LogThrow("Spline index invalid");
  }
  if (GetSplinesUsed() <= sIndex) {
    LogThrow("Spline index invalid");
  }
  int k = fSplineIndex->hostPtr()[sIndex+1]-fSplineIndex->hostPtr()[sIndex]-2;
  return k/2;
}

double Cache::Weight::UniformSpline::GetSplineLowerBound(int sIndex) {
  if (sIndex < 0) {
    LogThrow("Spline index invalid");
  }
  if (GetSplinesUsed() <= sIndex) {
    LogThrow("Spline index invalid");
  }
  int knotsIndex = fSplineIndex->hostPtr()[sIndex];
  return fSplineSpace->hostPtr()[knotsIndex];
}

double Cache::Weight::UniformSpline::GetSplineUpperBound(int sIndex) {
  if (sIndex < 0) {
    LogThrow("Spline index invalid");
  }
  if (GetSplinesUsed() <= sIndex) {
    LogThrow("Spline index invalid");
  }
  int knotCount = GetSplineKnotCount(sIndex);
  double lower = GetSplineLowerBound(sIndex);
  int knotsIndex = fSplineIndex->hostPtr()[sIndex];
  double step = fSplineSpace->hostPtr()[knotsIndex+1];
  return lower + (knotCount-1)/step;
}

double Cache::Weight::UniformSpline::GetSplineLowerClamp(int sIndex) {
  int i = GetSplineParameterIndex(sIndex);
  if (i<0) {
    LogThrow("Spline lower clamp index out of bounds");
  }
  if (fLowerClamp.size() <= i) {
    LogThrow("Spline lower clamp index out of bounds");
  }
  return fLowerClamp.hostPtr()[i];
}

double Cache::Weight::UniformSpline::GetSplineUpperClamp(int sIndex) {
  int i = GetSplineParameterIndex(sIndex);
  if (i<0) {
    LogThrow("Spline upper clamp index out of bounds");
  }
  if (fUpperClamp.size() <= i) {
    LogThrow("Spline upper clamp index out of bounds");
  }
  return fUpperClamp.hostPtr()[i];
}

double Cache::Weight::UniformSpline::GetSplineKnotValue(int sIndex, int knot) {
  if (sIndex < 0) {
    LogThrow("Spline index invalid");
  }
  if (GetSplinesUsed() <= sIndex) {
    LogThrow("Spline index invalid");
  }
  int knotsIndex = fSplineIndex->hostPtr()[sIndex];
  int count = GetSplineKnotCount(sIndex);
  if (knot < 0) {
    LogThrow("Knot index invalid");
  }
  if (count <= knot) {
    LogThrow("Knot index invalid");
  }
  return fSplineSpace->hostPtr()[knotsIndex+2+2*knot];
}

double Cache::Weight::UniformSpline::GetSplineKnotSlope(int sIndex, int knot) {
  if (sIndex < 0) {
    LogThrow("Spline index invalid");
  }
  if (GetSplinesUsed() <= sIndex) {
    LogThrow("Spline index invalid");
  }
  int knotsIndex = fSplineIndex->hostPtr()[sIndex];
  int count = GetSplineKnotCount(sIndex);
  if (knot < 0) {
    LogThrow("Knot index invalid");
  }
  if (count <= knot) {
    LogThrow("Knot index invalid");
  }
  return fSplineSpace->hostPtr()[knotsIndex+2+2*knot+1];
}

#include "CacheAtomicMult.h"
#include "CalculateUniformSpline.h"

// Define CACHE_DEBUG to get lots of output from the host
#undef CACHE_DEBUG
#define PRINT_STEP 3
namespace {
  // A function to be used as the kernel on either the CPU or GPU.  This
  // must be valid CUDA coda.
  HEMI_KERNEL_FUNCTION(HEMISplinesKernel,
                       double* results,
                       const double* params,
                       const double* lowerClamp,
                       const double* upperClamp,
                       const WEIGHT_BUFFER_FLOAT* knots,
                       const int* rIndex,
                       const short* pIndex,
                       const int* sIndex,
                       const int NP) {
    for (int i : hemi::grid_stride_range(0,NP)) {
      const int id0 = sIndex[i];
      const int id1 = sIndex[i+1];
      const int dim = id1-id0;
      const double x = params[pIndex[i]];
      const double lClamp = lowerClamp[pIndex[i]];
      const double uClamp = upperClamp[pIndex[i]];

      double v = CalculateUniformSpline(x,
                                        lClamp, uClamp,
                                        &knots[id0],dim);

#ifdef CACHE_DEBUG
      #ifndef HEMI_DEV_CODE
            if (rIndex[i] < PRINT_STEP) {
                double step = 1.0/knots[id0+1];
                LogInfo << "CACHE_DEBUG: uniform " << i
                        << " iEvt " << rIndex[i]
                        << " iPar " << pIndex[i]
                        << " = " << params[pIndex[i]]
                        << " m " << knots[id0] << " d "  << knots[id0+1]
                        << " s " << step
                        << " --> " << v
                        << " l: " << lClamp
                        << " u: " << uClamp
                        << " d: " << dim
                       << std::endl;
                for (int k = 0; k < (dim-2)/2; ++k) {
                    LogInfo << "CACHE_DEBUG:     " << k
                           << " x: " << knots[id0] + k*step
                           << " y: " << knots[id0+2+2*k]
                           << " m: " << knots[id0+2+2*k+1]
                           << std::endl;
                }
            }
#endif
#endif

      CacheAtomicMult(&results[rIndex[i]], v);
    }
  }
}

void Cache::Weight::UniformSpline::Reset() {
  // Use the parent reset.
  Cache::Weight::Base::Reset();
  // Reset this class
  fSplinesUsed = 0;
  fSplineSpaceUsed = 0;
}

bool Cache::Weight::UniformSpline::Apply() {
  if (GetSplinesUsed() < 1) return false;

  HEMISplinesKernel splinesKernel;
  hemi::launch(splinesKernel,
               fWeights.writeOnlyPtr(),
               fParameters.readOnlyPtr(),
               fLowerClamp.readOnlyPtr(),
               fUpperClamp.readOnlyPtr(),
               fSplineSpace->readOnlyPtr(),
               fSplineResult->readOnlyPtr(),
               fSplineParameter->readOnlyPtr(),
               fSplineIndex->readOnlyPtr(),
               GetSplinesUsed()
  );

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
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
