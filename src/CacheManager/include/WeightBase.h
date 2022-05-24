#ifndef WeightBase_hxx_seen
#define WeightBase_hxx_seen

#include "CacheWeights.h"
#include "CacheParameters.h"

#include "hemi/array.h"

#include <string>

namespace Cache {
    namespace Weight {
        class Base;
    }
}

// Do a definition here to "trick" nvcc which doesn't like the type to be
// typedef'ed
#define WEIGHT_BUFFER_FLOAT double

/// A base class for the weight calculators.  This holds the pointer to the
/// weights being accumulated, the input parameter values, and the name of the
/// weight calculator.
class Cache::Weight::Base {
public:
    // Construct the class.  This should allocate all the memory on the host
    // and on the GPU.  The normalizations are applied to the event weights
    // which are managed by the EventWeights class.
    Base(std::string name,
         Cache::Weights::Results& weights,
         Cache::Parameters::Values& parameters)
        : fName(name), fWeights(weights), fParameters(parameters) {}

    // Deconstruct the class.  This should deallocate all the memory
    // everyplace.
    virtual ~Base() {}

    /// Apply the weights to the event weight cache.  This will run a kernel
    /// to modify the weights cache.
    virtual bool Apply() = 0;

    std::size_t GetResidentMemory() {return fTotalBytes;}

    std::string GetName() {return fName;}

protected:

    // The name of the weight calculator.
    std::string fName;

    // Save the event weight cache reference for later use
    Cache::Weights::Results& fWeights;

    // Save the parameter cache reference for later use
    Cache::Parameters::Values& fParameters;

    std::size_t fTotalBytes{0};

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
