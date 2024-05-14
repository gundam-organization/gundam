#include "CacheParameters.h"

#include "ParameterSet.h"
#include "GundamGlobals.h"

#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"

#include <vector>
#include <set>

LoggerInit([]{
  Logger::setUserHeaderStr("[Cache::Parameters]");
});

bool Cache::Parameters::UsingCUDA() {
#ifdef __CUDACC__
    return true;
#else
    return false;
#endif
}

Cache::Parameters::Parameters(std::size_t parameters)
: fParameterCount{parameters} {
    LogInfo << "Cached Parameters -- input parameter count: "
            << GetParameterCount()
            << std::endl;

    fTotalBytes = 0;
    fTotalBytes += GetParameterCount()*sizeof(double); // fParameters
    fTotalBytes += GetParameterCount()*sizeof(double);  // fLowerClamp
    fTotalBytes += GetParameterCount()*sizeof(double);  // fUpperclamp

    try {
        // The mirrors are only on the CPU, so use vectors.  Initialize with
        // lowest and highest floating point values.
        fLowerMirror.reset(new std::vector<double>(
                               GetParameterCount(),
                               std::numeric_limits<double>::lowest()));
        LogThrowIf(not fLowerMirror, "Bad LowerMirror alloc");
        fUpperMirror.reset(new std::vector<double>(
                               GetParameterCount(),
                               std::numeric_limits<double>::max()));
        LogThrowIf(not fUpperMirror, "Bad UpperMirror alloc");

        // Get CPU/GPU memory for the parameter values.  The mirroring is done
        // to every entry, so its also done on the GPU.  The parameters are
        // copied every iteration, so pin the CPU memory into the page set.
        fParameters.reset(new hemi::Array<double>(GetParameterCount()));
        LogThrowIf(not fParameters, "Bad Parameters alloc");
        fLowerClamp.reset(new hemi::Array<double>(GetParameterCount(),false));
        LogThrowIf(not fLowerClamp, "Bad LowerClamp alloc");
        fUpperClamp.reset(new hemi::Array<double>(GetParameterCount(),false));
        LogThrowIf(not fUpperClamp, "Bad UpperClamp alloc");
    }
    catch (...) {
        LogError << "Failed to allocate memory, so stopping" << std::endl;
        LogThrow("Not enough memory available");
    }

    // Initialize the caches.  Don't try to zero everything since the
    // caches can be huge.
    Reset();
}

Cache::Parameters::~Parameters() {}

void Cache::Parameters::Reset() {
    std::fill(fLowerClamp->hostPtr(),
              fLowerClamp->hostPtr() + GetParameterCount(),
              std::numeric_limits<double>::lowest());
    std::fill(fUpperClamp->hostPtr(),
              fUpperClamp->hostPtr() + GetParameterCount(),
              std::numeric_limits<double>::max());
}

double Cache::Parameters::GetParameter(int parIdx) const {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    return fParameters->hostPtr()[parIdx];
}

void Cache::Parameters::SetParameter(int parIdx, double value) {
    LogThrowIf((parIdx < 0), "Parameter index out of range");
    if (GetParameterCount() <= parIdx) throw;
    LogThrowIf((GetParameterCount() <= parIdx), "Parameter index out of range");
    double lm = fLowerMirror->at(parIdx);
    double um = fUpperMirror->at(parIdx);
    // Mirror the input value at lm and um.
    int brake = 20;
    while (value < lm || value > um) {
        if (value < lm) value = lm + (lm - value);
        if (value > um) value = um - (value - um);
        if (--brake < 1) throw;
    }
    fParameters->hostPtr()[parIdx] = value;
}

double Cache::Parameters::GetLowerMirror(int parIdx) {
    LogThrowIf((parIdx < 0), "Parameter index out of range");
    LogThrowIf((GetParameterCount() <= parIdx), "Parameter index out of range");
    return fLowerMirror->at(parIdx);
}

void Cache::Parameters::SetLowerMirror(int parIdx, double value) {
    LogThrowIf((parIdx < 0), "Parameter index out of range");
    LogThrowIf((GetParameterCount() <= parIdx), "Parameter index out of range");
    fLowerMirror->at(parIdx) = value;
}

double Cache::Parameters::GetUpperMirror(int parIdx) {
    LogThrowIf((parIdx < 0), "Parameter index out of range");
    LogThrowIf((GetParameterCount() <= parIdx), "Parameter index out of range");
    return fUpperMirror->at(parIdx);
}

void Cache::Parameters::SetUpperMirror(int parIdx, double value) {
    LogThrowIf((parIdx < 0), "Parameter index out of range");
    LogThrowIf((GetParameterCount() <= parIdx), "Parameter index out of range");
    fUpperMirror->at(parIdx) = value;
}

double Cache::Parameters::GetLowerClamp(int parIdx) {
    LogThrowIf((parIdx < 0), "Parameter index out of range");
    LogThrowIf((GetParameterCount() <= parIdx), "Parameter index out of range");
    return fLowerClamp->hostPtr()[parIdx];
}

void Cache::Parameters::SetLowerClamp(int parIdx, double value) {
    LogThrowIf((parIdx < 0), "Parameter index out of range");
    LogThrowIf((GetParameterCount() <= parIdx), "Parameter index out of range");
    fLowerClamp->hostPtr()[parIdx] = value;
}

double Cache::Parameters::GetUpperClamp(int parIdx) {
    LogThrowIf((parIdx < 0), "Parameter index out of range");
    LogThrowIf((GetParameterCount() <= parIdx), "Parameter index out of range");
    return fUpperClamp->hostPtr()[parIdx];
}

void Cache::Parameters::SetUpperClamp(int parIdx, double value) {
    LogThrowIf((parIdx < 0), "Parameter index out of range");
    LogThrowIf((GetParameterCount() <= parIdx), "Parameter index out of range");
    fUpperClamp->hostPtr()[parIdx] = value;
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
