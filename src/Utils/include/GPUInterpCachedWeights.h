#ifndef GPUInterpCachedWeights_hxx_seen
#define GPUInterpCachedWeights_hxx_seen

#include "hemi/array.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace GPUInterp {
    class CachedWeights;
};

/// A class to calculate and cache a bunch of events weights.
class GPUInterp::CachedWeights {
private:
    static CachedWeights* fSingleton;  // You get one guess...

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
    static CachedWeights* Get() {return fSingleton;}
    static CachedWeights* Create(std::size_t results,
                                 std::size_t parameters,
                                 std::size_t norms,
                                 std::size_t splines,
                                 std::size_t knots) {
        if (!fSingleton) {
            fSingleton = new CachedWeights(
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
    CachedWeights(std::size_t results,
                  std::size_t parameters,
                  std::size_t norms,
                  std::size_t splines,
                  std::size_t knots);

    // Deconstruct the class.  This should deallocate all the memory
    // everyplace.
    ~CachedWeights();

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

    /// Set the parameter value for index i ib the host memory.  This will
    /// invalidate the fParameters on the host device.
    void SetParameter(int parIdx, double value);

    void SetLowerMirror(int parIdx, double value);
    void SetUpperMirror(int parIdx, double value);

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

};

#endif

// Local Variables:
// mode:c++
// c-basic-offset:4
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
