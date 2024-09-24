#include "CacheParameters.h"

#include "ParameterSet.h"
#include "GundamGlobals.h"

#include "Logger.h"

#include <vector>
#include <set>

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[Cache::Parameters]"); });
#endif

bool Cache::Parameters::UsingCUDA() {
#ifdef __CUDACC__
    return true;
#else
    return false;
#endif
}

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#endif

bool Cache::Parameters::HasGPU(bool dump) {
#ifndef __CUDACC__
    return false;
#else
    cudaError_t status;
    int devCount;
    status = cudaGetDeviceCount(&devCount);
    if (status != cudaSuccess) {
        if (dump) LogInfo << "CUDA DEVICE COUNT: No Device" << std::endl;
        return false;
    }

    int devId;
    status = cudaGetDevice(&devId);
    if (status != cudaSuccess) {
        if (dump) LogInfo << "CUDA DEVICE COUNT: No Device" << std::endl;
        return false;
    }

    cudaDeviceProp prop;
    status = cudaGetDeviceProperties(&prop, devId);
    if (status != cudaSuccess) {
        if (dump) LogInfo << "CUDA DEVICE COUNT: No Device" << std::endl;
        return false;
    }

    if (dump) {
        LogInfo << "CUDA DEVICE COUNT:         " << devCount << std::endl;
        LogInfo << "CUDA DEVICE ID:            " << devId << std::endl;
        LogInfo << "CUDA DEVICE NAME:          " << prop.name << std::endl;
        LogInfo << "CUDA COMPUTE CAPABILITY:   " << prop.major << "." << prop.minor << std::endl;
        LogInfo << "CUDA PROCESSORS:           " << prop.multiProcessorCount << std::endl;
        LogInfo << "CUDA PROCESSOR THREADS:    " << prop.maxThreadsPerMultiProcessor << std::endl;
        LogInfo << "CUDA MAX THREADS:          " << prop.maxThreadsPerMultiProcessor*prop.multiProcessorCount << std::endl;
        LogInfo << "CUDA THREADS PER BLOCK:    " << prop.maxThreadsPerBlock << std::endl;
        LogInfo << "CUDA BLOCK MAX DIM:        " << "X:" << prop.maxThreadsDim[0]
                  << " Y:" << prop.maxThreadsDim[1]
                  << " Z:" << prop.maxThreadsDim[2] << std::endl;
        LogInfo << "CUDA GRID MAX DIM:         " << "X:" << prop.maxGridSize[0]
                  << " Y:" << prop.maxGridSize[1]
                  << " Z:" << prop.maxGridSize[2] << std::endl;
        LogInfo << "CUDA WARP:                 " << prop.warpSize << std::endl;
        LogInfo << "CUDA CLOCK:                " << prop.clockRate << std::endl;
        LogInfo << "CUDA GLOBAL MEM:           " << prop.totalGlobalMem << std::endl;
        LogInfo << "CUDA SHARED MEM:           " << prop.sharedMemPerBlock << std::endl;
        LogInfo << "CUDA L2 CACHE MEM:         " << prop.l2CacheSize << std::endl;
        LogInfo << "CUDA CONST MEM:            " << prop.totalConstMem << std::endl;
        LogInfo << "CUDA MEM PITCH:            " << prop.memPitch << std::endl;
        LogInfo << "CUDA REGISTERS:            " << prop.regsPerBlock << std::endl;
    }
    if (prop.totalGlobalMem < 1) return false;
    if (prop.multiProcessorCount < 1) return false;
    if (prop.maxThreadsPerBlock < 1) return false;
    return true;
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
