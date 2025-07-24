#include "CacheWeights.h"
#include "WeightBase.h"
#include "WeightKriged.h"

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
Cache::Weight::Kriged::Kriged(
    Cache::Weights::Results& results,
    Cache::Parameters::Values& parameters,
    std::size_t dials,
    std::size_t weights,
    std::size_t tableSpace,
    const std::map<const std::vector<double>*, int>& tables)
    : Cache::Weight::Base("kriged",results,parameters),
      fReserved(dials), fUsed(0),
      fWeightsReserved(weights), fWeightsUsed(0),
      fTableReserved(tableSpace), fTableUsed(0),
      fTableOffsets(tables),
      fMinWeightsPerResult(0), fMaxWeightsPerResult(0),
      fSumWeightsPerResult(0), fSum2WeightsPerResult(0) {

    LogInfo << "Reserved " << GetName()
            << " Kriged Dials: " << GetReserved()
            << " Weight Space: " << GetWeightsReserved()
            << " (" << 1.0*GetWeightsReserved()/GetReserved() << " per dial)"
            << " Table Space: " << GetTableReserved()
            << " (" << 1.0*GetTableReserved()/GetReserved() << " per dial)"
            << std::endl;

    if (GetReserved() < 1) return;

    fTotalBytes += GetReserved()*sizeof(int);        // fResults
    fTotalBytes += GetReserved()*sizeof(int);        // fOffsets
    fTotalBytes += GetReserved()*sizeof(int);        // fIndices
    fTotalBytes += GetReserved()*sizeof(float);      // fConstants
    fTotalBytes += GetWeightsReserved()*sizeof(WEIGHT_BUFFER_FLOAT); // fTable;
    fTotalBytes += GetTableReserved()*sizeof(WEIGHT_BUFFER_FLOAT); // fTable;

    LogInfo << "  Approximate Memory Size for " << GetName()
            << ": " << GetResidentMemory()/1E+9
            << " GB" << std::endl;

    LogInfo << "    Tables " << std::endl;
    for (const auto& table : fTableOffsets) {
        LogInfo << "      Table: " << (void*) table.first
                << " Offset: " << table.second
                << " Size: " << table.first->size()
                << std::endl;
    }

    try {
        // Get the CPU/GPU memory for the spline index tables.  These are
        // copied once during initialization so do not pin the CPU memory into
        // the page set.
        fResults.reset(new hemi::Array<int>(GetReserved(),false));
        LogThrowIf(not fResults, "Bad Results alloc");
        LogThrowIf(fResults->size() != GetReserved(), "Incorrect result size");

        fOffsets.reset(new hemi::Array<int>(GetReserved(),false));
        LogThrowIf(not fOffsets, "Bad Offsets alloc");

        fIndices.reset(
            new hemi::Array<int>(GetWeightsReserved(),false));
        LogThrowIf(not fIndices, "Bad Indicies alloc");

        fConstants.reset(
            new hemi::Array<float>(GetWeightsReserved(),false));
        LogThrowIf(not fConstants, "Bad Constants alloc");

        // Get the CPU/GPU memory for the tables.  This is copied for each
        // evaluation.  The table is filled before this class is used.
        fTable.reset(
            new hemi::Array<WEIGHT_BUFFER_FLOAT>(fTableReserved,false));
        LogThrowIf(not fTable, "Bad Table alloc");
    }
    catch (...) {
        LogError << "Failed to allocate memory, so stopping" << std::endl;
        LogThrow("Not enough memory available");
    }

    // Initialize the caches.  Don't try to zero everything since the
    // caches can be huge.
    Reset();
    fResults->hostPtr()[0] = 0;
    fOffsets->hostPtr()[0] = 0;
    fIndices->hostPtr()[0] = 0;
    fConstants->hostPtr()[0] = 0;
    fTable->hostPtr()[0] = 0;
}

void Cache::Weight::Kriged::AddData(
    int resIndex,
    const std::vector<double>* table,
    const std::vector<std::pair<int,float>>& weights) {

    if (resIndex < 0) {
        LogError << "Invalid result index"
                 << std::endl;
        LogExit("Negative result index");
    }

    if (fWeights.size() <= resIndex) {
        LogError << "Invalid result index"
                 << fWeights.size() << " <= " << resIndex
                 << std::endl;
        LogExit("Result index out of bounds");
    }

    int newIndex = fUsed++;
    if (fUsed > fReserved) {
        LogError << "Not enough space reserved for dials"
                 << fUsed << " < " << fReserved
                 << std::endl;
        LogExit("Not enough space reserved for dials");
    }

    if (fResults->size() <= newIndex) {
        LogError << "Invalid new index"
                 << fResults->size() << " <= " << newIndex
                 << std::endl;
        LogExit("Result index out of bounds");
    }

    auto tableEntry = fTableOffsets.find(table);
    if (tableEntry == fTableOffsets.end()) {
        LogError << "Request to create Kriged weight for invalid table"
                 << std::endl;
        LogExit("Invalid table request");
    }

    if (fMinWeightsPerResult == 0 || fMinWeightsPerResult > weights.size()) {
        fMinWeightsPerResult = weights.size();
    }
    if (fMaxWeightsPerResult < weights.size()) {
        fMaxWeightsPerResult = weights.size();
    }
    fSumWeightsPerResult += weights.size();
    fSum2WeightsPerResult += weights.size()*weights.size();

    fResults->hostPtr()[newIndex] = resIndex;
    // Fill the offsets, indices, and weights here.
    for (const std::pair<int,float>& weight: weights) {
        int offset = tableEntry->second + weight.first;
        if (fTable->size() <= offset) {
            LogError << "Insufficent table space"
                     << fTable->size() << " <= " << offset
                     << std::endl;
            LogExit("Insufficient table space");
        }
        if (fWeightsUsed >= fWeightsReserved) {
            LogError << "Weights used more than reserved "
                     << fWeightsUsed << " >= " << fWeightsReserved
                     << std::endl;
            LogExit("Invalid fWeightsUsed");
        }
        fIndices->hostPtr()[fWeightsUsed] = offset;
        fConstants->hostPtr()[fWeightsUsed] = weight.second;
        fOffsets->hostPtr()[newIndex] = ++fWeightsUsed;
    }
}

#include "CacheAtomicMult.h"

namespace {
    // A function to be used as the kernel on either the CPU or GPU.  This
    // must be valid CUDA coda.
    HEMI_KERNEL_FUNCTION(HEMIKrigedKernel,
                         double* results,
                         const int* rIndex,
                         const int* offsets,
                         const int* indices,
                         const float* weights,
                         const WEIGHT_BUFFER_FLOAT* dataTable,
                         const int nDials) {

        for (int i : hemi::grid_stride_range(0,nDials)) {
            int begin = i>0 ? offsets[i-1]: 0;
            const int end = offsets[i];
            double v = 0;
            // Oh noes!  An internal loop!  Each event might have a different
            // number of kriging weights, but the maximum number of weights
            // per dial should (hopefully) be pretty small, and most dials
            // should (mostly) have the same number of weights. That is the
            // hope anyway.
            while (begin < end) {
                v += dataTable[indices[begin]]*weights[begin];
                ++begin;
            }
            CacheAtomicMult(&results[rIndex[i]], v);
        }
    }
}

void Cache::Weight::Kriged::Reset() {
    // Use the parent reset.
    Cache::Weight::Base::Reset();
    // Reset this class
    fUsed = 0;
    fTableUsed = 0;
}

bool Cache::Weight::Kriged::Apply() {
    if (GetUsed() < 1) return false;

    // Fill the tables.
    for (const auto& table : fTableOffsets) {
        int offset = table.second;
        for (const double entry : (*table.first)) {
            LogThrowIf(not std::isfinite(entry),"Table entry is not finite");
            fTable->hostPtr()[offset++] = entry;
        }
    }

    HEMIKrigedKernel krigedKernel;
    hemi::launch(krigedKernel,
                 fWeights.ptr(),
                 fResults->readOnlyPtr(),
                 fOffsets->readOnlyPtr(),
                 fIndices->readOnlyPtr(),
                 fConstants->readOnlyPtr(),
                 fTable->readOnlyPtr(),
                 GetUsed()
        );

    return true;
}

std::string Cache::Weight::Kriged::DumpSummary() const {
    std::ostringstream out;
    double norm = fResults->size();
    double avg = fSumWeightsPerResult / norm;
    double sig = fSum2WeightsPerResult / norm;
    sig = std::sqrt(sig - avg*avg);

    out << "Krige Summary: "
        << "Min/Max: " << fMinWeightsPerResult
        << "/" << fMaxWeightsPerResult
        << " Average: " << avg << "+/-" << sig;

    return out.str();
}

// An MIT Style License

// Copyright (c) 2024 Gundam Developers

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
