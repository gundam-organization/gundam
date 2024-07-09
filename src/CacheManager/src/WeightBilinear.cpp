#include "CacheWeights.h"
#include "WeightBase.h"
#include "WeightBilinear.h"

#include <algorithm>
#include <iostream>
#include <exception>
#include <limits>
#include <cmath>

#include <hemi/hemi_error.h>
#include <hemi/launch.h>
#include <hemi/grid_stride_range.h>

#include "Logger.h"
LoggerInit([]{
  Logger::setUserHeaderStr("[Cache::Weight::Bilinear]");
});

// The constructor
Cache::Weight::Bilinear::Bilinear(
    Cache::Weights::Results& weights,
    Cache::Parameters::Values& parameters,
    Cache::Parameters::Clamps& lowerClamps,
    Cache::Parameters::Clamps& upperClamps,
    std::size_t splines, std::size_t space)
    : Cache::Weight::Base("bilinear",weights,parameters),
      fLowerClamp(lowerClamps), fUpperClamp(upperClamps),
      fReserved(splines), fUsed(0),
      fSpaceReserved(space), fSpaceUsed(0) {

    LogInfo << "Reserved " << GetName() << " Splines: "
            << GetReserved() << std::endl;
    if (GetReserved() < 1) return;

    fTotalBytes += GetReserved()*sizeof(int);        // fResult
    fTotalBytes += 2*GetReserved()*sizeof(short);    // fParameter
    fTotalBytes += (1+GetReserved())*sizeof(int);    // fIndex

    LogInfo << "Reserved " << GetName()
            << " Spline Space: " << GetSpaceReserved()
            << std::endl;
    fTotalBytes += GetSpaceReserved()*sizeof(WEIGHT_BUFFER_FLOAT);  // fData

    LogInfo << "Approximate Memory Size for " << GetName()
            << ": " << fTotalBytes/1E+9
            << " GB" << std::endl;

    try {
        // Get the CPU/GPU memory for the spline index tables.  These are
        // copied once during initialization so do not pin the CPU memory into
        // the page set.
        fResult.reset(new hemi::Array<int>(GetReserved(),false));
        LogThrowIf(not fResult, "Bad SplineResult alloc");
        fParameter.reset(
            new hemi::Array<short>(2*GetReserved(),false));
        LogThrowIf(not fParameter, "Bad SplineParameter alloc");
        fIndex.reset(new hemi::Array<int>(1+GetReserved(),false));
        LogThrowIf(not fIndex, "Bad SplineIndex alloc");

        // Get the CPU/GPU memory for the spline knots.  This is copied once
        // during initialization so do not pin the CPU memory into the page
        // set.
        fData.reset(
            new hemi::Array<WEIGHT_BUFFER_FLOAT>(GetSpaceReserved(),false));
        LogThrowIf(not fData, "Bad SplineSpacealloc");
    }
    catch (...) {
        LogError << "Failed to allocate memory, so stopping" << std::endl;
        LogThrow("Not enough memory available");
    }

    // Initialize the caches.  Don't try to zero everything since the
    // caches can be huge.
    Reset();
    fIndex->hostPtr()[0] = 0;
}

// The destructor
Cache::Weight::Bilinear::~Bilinear() {}

void Cache::Weight::Bilinear::AddData(int resIndex,
                                        int par1Index, int par2Index,
                                        const std::vector<double>& data) {
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
    if (par1Index < 0) {
        LogError << "Invalid parameter 1 index"
               << std::endl;
        LogThrow("Negative parameter 1 index");
    }
    if (fParameters.size() <= par1Index) {
        LogError << "Invalid parameter 1 index " << par1Index
               << std::endl;
        LogThrow("Parameter 1 index out of bounds");
    }
    if (par2Index < 0) {
        LogError << "Invalid parameter 2 index"
               << std::endl;
        LogThrow("Negative parameter 2 index");
    }
    if (fParameters.size() <= par2Index) {
        LogError << "Invalid parameter 2 index " << par2Index
               << std::endl;
        LogThrow("Parameter 2 index out of bounds");
    }
    if (data.size() < 11) {
        LogError << "Insufficient points in spline " << data.size()
               << std::endl;
        LogThrow("Invalid number of spline points");
    }

    int newIndex = fUsed++;
    if (fUsed > fReserved) {
        LogError << "Not enough space reserved for splines"
                  << std::endl;
        LogThrow("Not enough space reserved for splines");
    }
    if (fSpaceUsed + data.size() > fSpaceReserved) {
        LogError << "Not enough space reserved for spline knots"
               << std::endl;
        LogThrow("Not enough space reserved for spline knots");
    }
    fResult->hostPtr()[newIndex] = resIndex;
    fParameter->hostPtr()[2*newIndex] = par1Index;
    fParameter->hostPtr()[2*newIndex+1] = par2Index;
    fIndex->hostPtr()[newIndex] = fSpaceUsed;
    for (double d : data) {
        fData->hostPtr()[fSpaceUsed++] = d;
        if (fSpaceUsed > fSpaceReserved) {
            LogError << "Not enough space reserved for spline knots"
                     << std::endl;
            LogThrow("Not enough space reserved for spline knots");
        }
    }
}

#include "CalculateBilinearInterpolation.h"
#include "CacheAtomicMult.h"

namespace {

    // A function to be used as the kernel on either the CPU or GPU.  This
    // must be valid CUDA coda.
    HEMI_KERNEL_FUNCTION(HEMIBilinearKernel,
                         double* results,
                         const double* params,
                         const double* lowerClamp,
                         const double* upperClamp,
                         const WEIGHT_BUFFER_FLOAT* dataTable,
                         const int* rIndex,
                         const short* pIndex,
                         const int* sIndex,
                         const int nData) {

        for (int i : hemi::grid_stride_range(0,nData)) {
            const int id0 = sIndex[i];
            const double x = params[pIndex[2*i]];
            const double y = params[pIndex[2*i+1]];
            const double lClamp = lowerClamp[pIndex[i]];
            const double uClamp = upperClamp[pIndex[i]];
            const WEIGHT_BUFFER_FLOAT* data = dataTable + id0;
            const int nx = *(data++);
            const int ny = *(data++);
            const double* xx = data; data += nx;
            const double* yy = data; data += ny;
            const double* knots = data;

            double v = CalculateBilinearInterpolation(
                x, y, lClamp, uClamp, knots, nx, ny, xx, nx, yy, ny);

            CacheAtomicMult(&results[rIndex[i]], v);
        }
    }
}

void Cache::Weight::Bilinear::Reset() {
    // Use the parent reset.
    Cache::Weight::Base::Reset();
    // Reset this class
    fUsed = 0;
    fSpaceUsed = 0;
}

bool Cache::Weight::Bilinear::Apply() {
    if (GetUsed() < 1) return false;

    HEMIBilinearKernel bilinearKernel;
    hemi::launch(bilinearKernel,
                 fWeights.writeOnlyPtr(),
                 fParameters.readOnlyPtr(),
                 fLowerClamp.readOnlyPtr(),
                 fUpperClamp.readOnlyPtr(),
                 fData->readOnlyPtr(),
                 fResult->readOnlyPtr(),
                 fParameter->readOnlyPtr(),
                 fIndex->readOnlyPtr(),
                 GetUsed()
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
// End:
