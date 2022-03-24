#ifndef CacheWeights_hxx_seen
#define CacheWeights_hxx_seen

#include "CacheParameters.h"

#include "hemi/array.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Cache {
    class Weights;
    namespace Weight {
        class Base;
    }
};

/// A class to calculate and cache a bunch of events weights.
class Cache::Weights {
public:
    typedef hemi::Array<double> Results;

private:
    // Save the event weight cache reference for later use
    Cache::Parameters::Values& fParameters;
    Cache::Parameters::Clamps& fLowerClamp;
    Cache::Parameters::Clamps& fUpperClamp;

    /// The (approximate) amount of memory required on the GPU.
    std::size_t fTotalBytes;

    /// An array of the calculated results that will be copied out of the
    /// class.  This is copied from the GPU to the CPU once per iteration.
    std::size_t    fResultCount;
    std::unique_ptr<Results> fResults;

    /// An array of the initial value for each result.  It's copied from the
    /// CPU to the GPU once at the beginning.
    std::unique_ptr<hemi::Array<double>> fInitialValues;

    /// An array of pointers to objects that will calculate the weights
    /// (e.g. WeightNormalization and WeightUniformSpline).  The objects are
    /// NOT owned by this array (they are owned by a unique_ptr's elsewyr in
    /// the code), but pointers are needed for efficiency.
    int fWeightCalculators{0};
    std::array<Cache::Weight::Base*,5> fWeightCalculator;

public:
    // Construct the class.  This should allocate all the memory on the host
    // and on the GPU.  The "results" are the total number of results to be
    // calculated (one result per event, often >1E+6).  The "parameters" are
    // the number of input parameters that are used (often ~1000).  The norms
    // are the total number of normalization parameters (typically a few per
    // event) used to calculate the results.  The splines are the total number
    // of spline parameters (with uniform not spacing) used to calculate the
    // results (typically a few per event).  The knots are the total number of
    // knots in all of the uniform splines (e.g. For 1000 splines with 7
    // knots for each spline, knots is 7000).
    Weights(Cache::Parameters::Values& parameters,
            Cache::Parameters::Clamps& lowerClamp,
            Cache::Parameters::Clamps& upperClamp,
            std::size_t results);

    // Deconstruct the class.  This should deallocate all the memory
    // everyplace.
    ~Weights();

    Results& GetWeights() {return *fResults;}

    /// Return the approximate allocated memory (e.g. on the GPU).
    std::size_t GetResidentMemory() const {return fTotalBytes;}

    /// Return the number of results that are used, and will be returned.
    std::size_t GetResultCount() const {return fResultCount;}

    int AddWeightCalculator(Cache::Weight::Base* v) {
        int index = fWeightCalculators++;
        fWeightCalculator.at(index) = v;
        return index;
    }

    /// Calculate the results and save them for later use.  This copies the
    /// results from the GPU to the CPU.
    virtual bool Apply();

    /// Get the result for index i from host memory.  This will trigger copying
    /// the results from the device if that is necessary.
    double GetResult(int i) const;
    double* GetResultPointer(int i) const;

    /// Set the result for index i in the host memory.  The results are NEVER
    /// copied to the device, so this will be overwritten as soon as the
    /// results are updated.  This is here for debugging.
    void SetResult(int i, double v);

    /// Get/Set the initial value for result i.
    double  GetInitialValue(int i) const;
    void SetInitialValue(int i, double v);
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
