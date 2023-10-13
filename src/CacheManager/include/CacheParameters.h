#ifndef CacheParameters_h_seen
#define CacheParameters_h_seen

#include "hemi/array.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Cache {
    class Parameters;
}

class Parameter;

/// A cache to hold the function parameter values, and the clamp values on the
/// output weights.  This also manages any mirroring that needs to be done.
class Cache::Parameters {
public:
    typedef hemi::Array<double> Values;
    typedef hemi::Array<double> Clamps;

    // Returns true if this is compiled with a CUDA compiler
    static bool UsingCUDA();

    // This is a singleton, so the constructor is private.
    Parameters(std::size_t parameters);

    ~Parameters();

    /// Reinitialize the cache.  This puts it into a state to be refilled, but
    /// does not deallocate any memory.
    void Reset();

    /// The arrays of values mirrored on the CPU and GPU (e.g. hemi arrays).
    Values& GetParameters() {return *fParameters;}
    Clamps& GetLowerClamps() {return *fLowerClamp;}
    Clamps& GetUpperClamps() {return *fUpperClamp;}

    /// Return the approximate allocated memory (e.g. on the GPU).
    std::size_t GetResidentMemory() const {return fTotalBytes;}

    /// Return the number of independent parameters.
    std::size_t GetParameterCount() const {return fParameterCount;}

    /// Get the parameter value for index i from host memory.  This will
    /// trigger copying the results from the device if that is necessary.
    double GetParameter(int parIdx) const;

    /// Set the parameter value for index i in the host memory.  This will
    /// invalidate the fParameters on the device.
    void SetParameter(int parIdx, double value);

    /// Get the lower (upper) bound of the mirroring region for parameter
    /// index i in the host memory.
    double GetLowerMirror(int parIdx);
    double GetUpperMirror(int parIdx);

    /// Set the lower (upper) bound of the mirroring region for parameter
    /// index i in the host memory.  This will invalidate the mirrors on the
    /// device.
    void SetLowerMirror(int parIdx, double value);
    void SetUpperMirror(int parIdx, double value);

    /// Get the lower (upper) clamp for parameter index i in the host memory.
    double GetLowerClamp(int parIdx);
    double GetUpperClamp(int parIdx);

    /// Set the lower (upper) clamp for parameter index i in the host memory.
    /// This will invalidate the clamps on the device.
    void SetLowerClamp(int parIdx, double value);
    void SetUpperClamp(int parIdx, double value);

private:
    std::size_t fTotalBytes;

    /// The parameter values to be calculated for.  This is copied from the
    /// CPU to the GPU once per iteration.
    std::size_t fParameterCount;
    std::unique_ptr<Values> fParameters;

    /// The value of the parameter can run from +inf to -inf, but will be
    /// mirrored to be between the upper and lower mirrors.  These copied from
    /// the CPU to the GPU once, and are then constant.
    std::unique_ptr<std::vector<double>> fLowerMirror;
    std::unique_ptr<std::vector<double>> fUpperMirror;

    /// The minimum and maximum value of the result for this parameter.  These
    ///  copied from the CPU to the GPU once, and are then constant.
    std::unique_ptr<Clamps> fLowerClamp;
    std::unique_ptr<Clamps> fUpperClamp;
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
