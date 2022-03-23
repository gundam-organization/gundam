#ifndef CacheEventWeights_hxx_seen
#define CacheEventWeights_hxx_seen

#include "hemi/array.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Cache {
    class EventWeights;
};

/// A class to calculate and cache a bunch of events weights.
class Cache::EventWeights {
private:
    static EventWeights* fSingleton;  // You get one guess...

    /// The (approximate) amount of memory required on the GPU.
    std::size_t fTotalBytes;

    /// An array of the calculated results that will be copied out of the
    /// class.  This is copied from the GPU to the CPU once per iteration.
    std::size_t    fResultCount;
    std::unique_ptr<hemi::Array<double>> fResults;

    /// An array of the initial value for each result.  It's copied from the
    /// CPU to the GPU once at the beginning.
    std::unique_ptr<hemi::Array<double>> fInitialValues;

    /// The parameter values to be calculated for.  This is copied from the
    /// CPU to the GPU once per iteration.
    std::size_t fParameterCount;
    std::unique_ptr<hemi::Array<double>> fParameters;

    /// The value of the parameter can run from +inf to -inf, but will be
    /// mirrored to be between the upper and lower mirrors.  These copied from
    /// the CPU to the GPU once, and are then constant.
    std::unique_ptr<std::vector<double>> fLowerMirror;
    std::unique_ptr<std::vector<double>> fUpperMirror;

    /// The minimum and maximum value of the result for this parameter.  These
    ///  copied from the CPU to the GPU once, and are then constant.
    std::unique_ptr<hemi::Array<float>> fLowerClamp;
    std::unique_ptr<hemi::Array<float>> fUpperClamp;

    ///////////////////////////////////////////////////////////////////////
    /// An array of indices into the results that go for each normalization.
    /// This is copied from the CPU to the GPU once, and is then constant.
    std::size_t fNormsReserved;
    std::size_t fNormsUsed;
    std::unique_ptr<hemi::Array<int>> fNormResult;

    /// An array of indices into the parameters that go for each
    /// normalization.  This is copied from the CPU to the GPU once, and is
    /// then constant.
    std::unique_ptr<hemi::Array<short>> fNormParameter;

    ///////////////////////////////////////////////////////////////////////
    /// An array of indices into the results that go for each spline.
    /// This is copied from the CPU to the GPU once, and is then constant.
    std::size_t fSplinesReserved;
    std::size_t fSplinesUsed;
    std::unique_ptr<hemi::Array<int>> fSplineResult;

    /// An array of indices into the parameters that go for each spline.  This
    /// is copied from the CPU to the GPU once, and is then constant.
    std::unique_ptr<hemi::Array<short>> fSplineParameter;

    /// An array of indices for the first knot of each spline.  This is copied
    /// from the CPU to the GPU once, and is then constant.
    std::unique_ptr<hemi::Array<int>> fSplineIndex;

    /// An array of the knots to calculate the splines.  This is copied from
    /// the CPU to the GPU once, and is then constant.
    std::size_t    fSplineKnotsReserved;
    std::size_t    fSplineKnotsUsed;
    std::unique_ptr<hemi::Array<float>> fSplineKnots;

public:
    static EventWeights* Get() {return fSingleton;}
    static EventWeights* Create(std::size_t results,
                                 std::size_t parameters,
                                 std::size_t norms,
                                 std::size_t splines,
                                 std::size_t knots) {
        if (!fSingleton) {
            fSingleton = new EventWeights(
                results,parameters,norms,splines,knots);
        }
        return fSingleton;
    }

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
    EventWeights(std::size_t results,
                  std::size_t parameters,
                  std::size_t norms,
                  std::size_t splines,
                  std::size_t knots);

    // Deconstruct the class.  This should deallocate all the memory
    // everyplace.
    ~EventWeights();

    /// Return the approximate allocated memory (e.g. on the GPU).
    std::size_t GetResidentMemory() const {return fTotalBytes;}

    /// Return the number of independent parameters.
    std::size_t GetParameterCount() const {return fParameterCount;}

    /// Return the number of results that are used, and will be returned.
    std::size_t GetResultCount() const {return fResultCount;}

    /// Return the number of normalization parameters that are reserved
    std::size_t GetNormsReserved() {return fNormsReserved;}

    /// Return the number of normalization parameters that are used.
    std::size_t GetNormsUsed() {return fNormsUsed;}

    /// Return the number of parameters using a spline with uniform knots that
    /// are reserved.
    std::size_t GetSplinesReserved() {return fSplinesReserved;}

    /// Return the number of parameters using a spline with uniform knots that
    /// are used.
    std::size_t GetSplinesUsed() {return fSplinesUsed;}

    /// Return the number of elements reserved to hold knots.
    std::size_t GetSplineKnotsReserved() const {return fSplineKnotsReserved;}

    /// Return the number of elements currently used to hold knots.
    std::size_t GetSplineKnotsUsed() const {return fSplineKnotsUsed;}

    /// Calculate the results and save them for later use.  This copies the
    /// results from the GPU to the CPU.
    void UpdateResults();

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

    /// Add a normalization parameter.
    int ReserveNorm(int resIndex, int parIndex);

    // Reserve space for a spline with uniform knot spacing.  This updates the
    // parameter index and data index arrays.  It returns the index for the
    // new spline.
    int ReserveSpline(int resIndex, int parIndex,
                      double low, double high, int points);

    // Add a new spline with uniform knot spacing to calculate a result. This
    // returns the index of the new spline, and fills the internal tables.
    int AddSpline(int resIndex, int parIndex,
                  double low, double high,
                  double points[], int nPoints);

    // Set a knot for a spline with uniform spacing.  This takes the index of
    // the spline that this spline will fill and the index of the control
    // point to be filled in that spline.  This can only be used after
    // ReserveSpline has been called.  The sIndex is the value returned by
    // ReserveSpline for the particular spline, and the kIndex is the knot
    // that will be filled.
    void SetSplineKnot(int sIndex, int kIndex, double value);

    // Get the index of the parameter for the spline at sIndex.
    int GetSplineParameterIndex(int sIndex);

    // Get the parameter value for the spline at sIndex.
    double GetSplineParameter(int sIndex);

    // Get the lower (upper) bound for the spline at sIndex.
    double GetSplineLowerBound(int sIndex);
    double GetSplineUpperBound(int sIndex);

    // Get the lower (upper) clamp for the spline at sIndex.
    double GetSplineLowerClamp(int sIndex);
    double GetSplineUpperClamp(int sIndex);

    // Get the number of knots in the spline at sIndex.
    int GetSplineKnotCount(int sIndex);

    // Get the function value for a knot in the spline at sIndex
    double GetSplineKnot(int sIndex,int knot);

    ////////////////////////////////////////////////////////////////////
    // This section is for the validation methods.  They should mostly be
    // NOOPs and should mostly not be called.
#undef GPUINTERP_SLOW_VALIDATION /* Define to have slow validations */

#ifdef GPUINTERP_SLOW_VALIDATION
    double GetSplineValue(int sIndex);

    /// An array of values for the result of each spline.  When this is
    /// active, it is filled but the kernel, but only copied to the CPU if
    /// it's access.  NOTE: Enabling this significantly slows the calculation
    /// since it adds another large copy from the GPU.
    std::unique_ptr<hemi::Array<double>> fSplineValue;
#endif

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
