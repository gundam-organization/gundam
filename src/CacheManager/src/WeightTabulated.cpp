#include "CacheWeights.h"
#include "WeightBase.h"
#include "WeightTabulated.h"

#include <algorithm>
#include <iostream>
#include <exception>
#include <limits>
#include <cmath>

#include <hemi/hemi_error.h>
#include <hemi/launch.h>
#include <hemi/grid_stride_range.h>

#include "Logger.h"

#ifndef DISABLE_USER_HEADER
LoggerInit([]{
  Logger::setUserHeaderStr("[Cache::Weight::Tabulated]");
});
#endif

// The constructor
Cache::Weight::Tabulated::Tabulated(
    Cache::Weights::Results& weights,
    Cache::Parameters::Values& parameters,
    std::size_t dials, std::size_t tableSpace,
    const std::map<const std::vector<double>*, int>& tables)
    : Cache::Weight::Base("tabulated",weights,parameters),
      fReserved(dials), fUsed(0),
      fDataReserved(tableSpace), fDataUsed(0),
      fTables(tables) {

    LogInfo << "Reserved " << GetName() << " Tabulated Dials: "
            << GetReserved() << std::endl;
    if (GetReserved() < 1) return;

    fTotalBytes += GetReserved()*sizeof(int);        // fResult
    fTotalBytes += GetReserved()*sizeof(int);        // fIndex
    fTotalBytes += GetReserved()*sizeof(float);      // fFraction
    fTotalBytes += GetDataReserved()*sizeof(WEIGHT_BUFFER_FLOAT); // fData;

    LogInfo << "Reserved " << GetName()
            << " Table Space: " << GetDataReserved()
            << std::endl;

    LogInfo << "Approximate Memory Size for " << GetName()
            << ": " << GetResidentMemory()/1E+9
            << " GB" << std::endl;

    try {
        // Get the CPU/GPU memory for the spline index tables.  These are
        // copied once during initialization so do not pin the CPU memory into
        // the page set.
        fResult.reset(new hemi::Array<int>(GetReserved(),false));
        LogThrowIf(not fResult, "Bad Result alloc");

        fIndex.reset(new hemi::Array<int>(GetReserved(),false));
        LogThrowIf(not fIndex, "Bad Index alloc");

        fFraction.reset(
            new hemi::Array<WEIGHT_BUFFER_FLOAT>(GetReserved(),false));
        LogThrowIf(not fFraction, "Bad Fraction alloc");

        // Get the CPU/GPU memory for the tables.  This is copied for each
        // evaluation.  The table is filled before this class is used.
        fData.reset(
            new hemi::Array<WEIGHT_BUFFER_FLOAT>(fDataReserved,false));
        LogThrowIf(not fData, "Bad Table alloc");
    }
    catch (...) {
        LogError << "Failed to allocate memory, so stopping" << std::endl;
        LogThrow("Not enough memory available");
    }

    // Initialize the caches.  Don't try to zero everything since the
    // caches can be huge.
    Reset();
    fResult->hostPtr()[0] = 0;
    fIndex->hostPtr()[0] = 0;
    fFraction->hostPtr()[0] = 0;
    fData->hostPtr()[0] = 0;
}

void Cache::Weight::Tabulated::AddData(int resIndex,
                                       const std::vector<double>* table,
                                       int index,
                                       double fraction) {

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

    int newIndex = fUsed++;
    if (fUsed > fReserved) {
        LogError << "Not enough space reserved for dials"
                  << std::endl;
        LogThrow("Not enough space reserved for dials");
    }

    auto tableEntry = fTables.find(table);
    if (tableEntry == fTables.end()) {
        LogError << "Request to create Tabulated weight for invalid table"
                 << std::endl;
        LogThrow("Invalid table request");
    }
    int offset = tableEntry->second + index;
    if (fData->size() <= offset) {
        LogError << "Insufficent table space" << std::endl;
        LogThrow("Insufficient table space");
    }

    fResult->hostPtr()[newIndex] = resIndex;
    fIndex->hostPtr()[newIndex] = offset;
    fFraction->hostPtr()[newIndex] = fraction;

}

#include "CacheAtomicMult.h"

namespace {

    // A function to be used as the kernel on either the CPU or GPU.  This
    // must be valid CUDA coda.
    HEMI_KERNEL_FUNCTION(HEMITabulatedKernel,
                         double* results,
                         const int* rIndex,
                         const int* index,
                         const WEIGHT_BUFFER_FLOAT* frac,
                         const WEIGHT_BUFFER_FLOAT* dataTable,
                         const int nData) {

        for (int i : hemi::grid_stride_range(0,nData)) {
            const WEIGHT_BUFFER_FLOAT* data = dataTable + index[i];
            const double f = frac[i];
            double v = (*data)*f;
            ++data;
            v += (*data)*(1.0-f);
            CacheAtomicMult(&results[rIndex[i]], v);
        }
    }
}

void Cache::Weight::Tabulated::Reset() {
    // Use the parent reset.
    Cache::Weight::Base::Reset();
    // Reset this class
    fUsed = 0;
    fDataUsed = 0;
}

bool Cache::Weight::Tabulated::Apply() {
    if (GetUsed() < 1) return false;

    // Fill the table.
    for (auto table : fTables) {
        int offset = table.second;
        for (double entry : (*table.first)) {
            fData->hostPtr()[offset++] = entry;
        }
    }

    HEMITabulatedKernel tabulatedKernel;
    hemi::launch(tabulatedKernel,
                 fWeights.writeOnlyPtr(),
                 fResult->readOnlyPtr(),
                 fIndex->readOnlyPtr(),
                 fFraction->readOnlyPtr(),
                 fData->readOnlyPtr(),
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
