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
        fUpperMirror.reset(new std::vector<double>(
                               GetParameterCount(),
                               std::numeric_limits<double>::max()));

        // Get CPU/GPU memory for the parameter values.  The mirroring is done
        // to every entry, so its also done on the GPU.  The parameters are
        // copied every iteration, so pin the CPU memory into the page set.
        fParameters.reset(new hemi::Array<double>(GetParameterCount()));
        fLowerClamp.reset(new hemi::Array<double>(GetParameterCount(),false));
        fUpperClamp.reset(new hemi::Array<double>(GetParameterCount(),false));
    }
    catch (...) {
        LogError << "Failed to allocate memory, so stopping" << std::endl;
        throw std::runtime_error("Not enough memory available");
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
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
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
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    return fLowerMirror->at(parIdx);
}

void Cache::Parameters::SetLowerMirror(int parIdx, double value) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    fLowerMirror->at(parIdx) = value;
}

double Cache::Parameters::GetUpperMirror(int parIdx) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    return fUpperMirror->at(parIdx);
}

void Cache::Parameters::SetUpperMirror(int parIdx, double value) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    fUpperMirror->at(parIdx) = value;
}

double Cache::Parameters::GetLowerClamp(int parIdx) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    return fLowerClamp->hostPtr()[parIdx];
}

void Cache::Parameters::SetLowerClamp(int parIdx, double value) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    fLowerClamp->hostPtr()[parIdx] = value;
}

double Cache::Parameters::GetUpperClamp(int parIdx) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    return fUpperClamp->hostPtr()[parIdx];
}

void Cache::Parameters::SetUpperClamp(int parIdx, double value) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
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
