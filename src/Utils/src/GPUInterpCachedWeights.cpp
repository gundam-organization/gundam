#include "GPUInterpCachedWeights.h"

#include <algorithm>
#include <iostream>
#include <exception>
#include <limits>

#include <hemi/hemi_error.h>
#include <hemi/launch.h>
#include <hemi/grid_stride_range.h>

#define INSIDE_GUNDAM
#ifdef INSIDE_GUNDAM
#include <Logger.h>
LoggerInit([](){
  Logger::setUserHeaderStr("[GPU]");
} )
#define LOGGER LogInfo
#endif

#ifndef LOGGER
#define LOGGER std::cout
#endif

GPUInterp::CachedWeights* GPUInterp::CachedWeights::fSingleton = nullptr;

// The constructor
GPUInterp::CachedWeights::CachedWeights(
    std::size_t results, std::size_t parameters, std::size_t norms,
    std::size_t splines, std::size_t knots)
    : fTotalBytes(0), fResultCount(results), fParameterCount(parameters),
      fNormsReserved(norms), fNormsUsed(0),
      fSplinesReserved(splines), fSplinesUsed(0),
      fSplineKnotsReserved(knots), fSplineKnotsUsed(0) {
    if (fResultCount<1) throw std::runtime_error("No results in weight cache");
    if (fParameterCount<1) throw std::runtime_error("No parameters");
    fSplineKnotsReserved = 2*fSplinesReserved + fSplineKnotsReserved;

    LOGGER << "Cached Weights: input parameter count: "
           << GetParameterCount()
           << std::endl;
    fTotalBytes += GetParameterCount()*sizeof(double); // fParameters
    fTotalBytes += GetParameterCount()*sizeof(float);  // fLowerClamp
    fTotalBytes += GetParameterCount()*sizeof(float);  // fUpperclamp

    LOGGER << "Cached Weights: output results: "
           << GetResultCount()
           << std::endl;
    fTotalBytes += GetResultCount()*sizeof(double);   // fResults
    fTotalBytes += GetResultCount()*sizeof(double);   // fInitialValues;

    LOGGER << "Cached Weights: reserved Normalizations: "
           << GetNormsReserved()
           << std::endl;
    fTotalBytes += GetNormsReserved()*sizeof(int);   // fNormResult
    fTotalBytes += GetNormsReserved()*sizeof(short); // fNormParameter

    LOGGER << "Reserved Splines: " << GetSplinesReserved() << std::endl;
    fTotalBytes += GetSplinesReserved()*sizeof(int);      // fSplineResult
    fTotalBytes += GetSplinesReserved()*sizeof(short);    // fSplineParameter
    fTotalBytes += (1+GetSplinesReserved())*sizeof(int);  // fSplineIndex

    LOGGER << "Reserved Spline Knots: " << GetSplineKnotsReserved()<< std::endl;
    fTotalBytes += GetSplineKnotsReserved()*sizeof(float);  // fSpineKnots

    LOGGER << "Approximate Memory Size: " << fTotalBytes/1E+9
              << " GB" << std::endl;

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
        fLowerClamp.reset(new hemi::Array<float>(GetParameterCount(),false));
        fUpperClamp.reset(new hemi::Array<float>(GetParameterCount(),false));

        // Get CPU/GPU memory for the results and thier initial values.  The
        // results are copied every time, so pin the CPU memory into the page
        // set.  The initial values are seldom changed, so they are not
        // pinned.
        fResults.reset(new hemi::Array<double>(GetResultCount()));
        fInitialValues.reset(new hemi::Array<double>(GetResultCount(),false));

        // Get the CPU/GPU memory for the normalization index tables.  These
        // are copied once during initialization so do not pin the CPU memory
        // into the page set.
        fNormResult.reset(new hemi::Array<int>(GetNormsReserved(),false));
        fNormParameter.reset(new hemi::Array<short>(GetNormsReserved(),false));

        // Get the CPU/GPU memory for the spline index tables.  These are
        // copied once during initialization so do not pin the CPU memory into
        // the page set.
        fSplineResult.reset(new hemi::Array<int>(GetSplinesReserved(),false));
        fSplineParameter.reset(
            new hemi::Array<short>(GetSplinesReserved(),false));
        fSplineIndex.reset(new hemi::Array<int>(1+GetSplinesReserved(),false));

        // Get the CPU/GPU memory for the spline knots.  This is copied once
        // during initialization so do not pin the CPU memory into the page
        // set.
        fSplineKnots.reset(
            new hemi::Array<float>(GetSplineKnotsReserved(),false));
    }
    catch (std::bad_alloc&) {
        LOGGER << "Failed to allocate memory, so stopping" << std::endl;
        throw std::runtime_error("Not enough memory available");
    }

    // Initialize the caches.  Don't try to zero everything since the
    // caches can be huge.
    fSplineIndex->hostPtr()[0] = 0;
    std::fill(fInitialValues->hostPtr(),
              fInitialValues->hostPtr() + fInitialValues->size(),
              1.0);
    std::fill(fLowerClamp->hostPtr(),
              fLowerClamp->hostPtr() + GetParameterCount(),
              std::numeric_limits<float>::lowest());
    std::fill(fUpperClamp->hostPtr(),
              fUpperClamp->hostPtr() + GetParameterCount(),
              std::numeric_limits<float>::max());

}

// The destructor
GPUInterp::CachedWeights::~CachedWeights() {}

double GPUInterp::CachedWeights::GetResult(int i) const {
    if (i < 0) throw;
    if (GetResultCount() <= i) throw;
    return fResults->hostPtr()[i];
}

void GPUInterp::CachedWeights::SetResult(int i, double v) {
    if (i < 0) throw;
    if (GetResultCount() <= i) throw;
    fResults->hostPtr()[i] = v;
}

double* GPUInterp::CachedWeights::GetResultPointer(int i) const {
    if (i < 0) throw;
    if (GetResultCount() <= i) throw;
    return (fResults->hostPtr() + i);
}

double  GPUInterp::CachedWeights::GetInitialValue(int i) const {
    if (i < 0) throw;
    if (GetResultCount() <= i) throw;
    return fInitialValues->hostPtr()[i];
}

void GPUInterp::CachedWeights::SetInitialValue(int i, double v) {
    if (i < 0) throw;
    if (GetResultCount() <= i) throw;
    fInitialValues->hostPtr()[i] = v;
}

double GPUInterp::CachedWeights::GetParameter(int parIdx) const {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    return fParameters->hostPtr()[parIdx];
}

void GPUInterp::CachedWeights::SetParameter(int parIdx, double value) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    fParameters->hostPtr()[parIdx] = value;
}

void GPUInterp::CachedWeights::SetLowerMirror(int parIdx, double value) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    fLowerMirror->at(parIdx) = value;
}

void GPUInterp::CachedWeights::SetUpperMirror(int parIdx, double value) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    fUpperMirror->at(parIdx) = value;
}

void GPUInterp::CachedWeights::SetLowerClamp(int parIdx, double value) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    fLowerClamp->hostPtr()[parIdx] = value;
}

void GPUInterp::CachedWeights::SetUpperClamp(int parIdx, double value) {
    if (parIdx < 0) throw;
    if (GetParameterCount() <= parIdx) throw;
    fUpperClamp->hostPtr()[parIdx] = value;
}

// Reserve space for another normalization parameter.
int GPUInterp::CachedWeights::ReserveNorm(int resIndex, int parIndex) {
    if (resIndex < 0) {
        LOGGER << "Invalid result index"
               << std::endl;
        throw std::runtime_error("Negative result index");
    }
    if (GetResultCount() <= resIndex) {
        LOGGER << "Invalid result index"
               << std::endl;
        throw std::runtime_error("Result index out of bounds");
    }
    if (parIndex < 0) {
        LOGGER << "Invalid parameter index"
               << std::endl;
        throw std::runtime_error("Negative parameter index");
    }
    if (GetParameterCount() <= parIndex) {
        LOGGER << "Invalid result index"
               << std::endl;
        throw std::runtime_error("Parameter index out of bounds");
    }
    int newIndex = fNormsUsed++;
    if (fNormsUsed > fNormsReserved) {
        LOGGER << "Not enough space reserved for Norms"
                  << std::endl;
        throw std::runtime_error("Not enough space reserved for results");
    }
    fNormResult->hostPtr()[newIndex] = resIndex;
    fNormParameter->hostPtr()[newIndex] = parIndex;
    return newIndex;
}

// Reserve space in the internal structures for spline with uniform knots.
// The knot values will still need to be set using set spline knot.
int GPUInterp::CachedWeights::ReserveSpline(
    int resIndex, int parIndex, double low, double high, int points) {
    if (resIndex < 0) {
        LOGGER << "Invalid result index"
               << std::endl;
        throw std::runtime_error("Negative result index");
    }
    if (GetResultCount() <= resIndex) {
        LOGGER << "Invalid result index"
               << std::endl;
        throw std::runtime_error("Result index out of bounds");
    }
    if (parIndex < 0) {
        LOGGER << "Invalid parameter index"
               << std::endl;
        throw std::runtime_error("Negative parameter index");
    }
    if (GetParameterCount() <= parIndex) {
        LOGGER << "Invalid result index"
               << std::endl;
        throw std::runtime_error("Parameter index out of bounds");
    }
    if (high <= low) {
        LOGGER << "Invalid spline bounds"
               << std::endl;
        throw std::runtime_error("Invalid spline bounds");
    }
    if (points < 3) {
        LOGGER << "Insufficient points in spline"
               << std::endl;
        throw std::runtime_error("Invalid number of spline points");
    }
    int newIndex = fSplinesUsed++;
    if (fSplinesUsed > fSplinesReserved) {
        LOGGER << "Not enough space reserved for splines"
                  << std::endl;
        throw std::runtime_error("Not enough space reserved for splines");
    }
    fSplineResult->hostPtr()[newIndex] = resIndex;
    fSplineParameter->hostPtr()[newIndex] = parIndex;
    if (fSplineIndex->hostPtr()[newIndex] != fSplineKnotsUsed) {
        LOGGER << "Last spline knot index should be at old end of splines"
                  << std::endl;
        throw std::runtime_error("Problem with control indices");
    }
    int knotIndex = fSplineKnotsUsed;
    fSplineKnotsUsed += 2; // Space for the upper and lower bound
    fSplineKnotsUsed += points; // Space for the knots.
    if (fSplineKnotsUsed > fSplineKnotsReserved) {
        LOGGER << "Not enough space reserved for spline knots"
               << std::endl;
        throw std::runtime_error("Not enough space reserved for spline knots");
    }
    fSplineIndex->hostPtr()[newIndex+1] = fSplineKnotsUsed;
    // Save values needed to calculate the spline offset index.  If the input
    // value is x, the index is
    // v = (x-CD[dataIndex])*CD[dataIndex+1].
    // i = v;
    // v = v - i;
    double invStep =  1.0*(points-1.0)/(high-low);
    fSplineKnots->hostPtr()[knotIndex] = low;
    fSplineKnots->hostPtr()[knotIndex+1] = invStep;

    return newIndex;
}

void GPUInterp::CachedWeights::SetSplineKnot(
    int sIndex, int kIndex, double value) {
    if (sIndex < 0) {
        LOGGER << "Requested spline index is negative"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    if (GetSplinesUsed() <= sIndex) {
        LOGGER << "Requested spline index is to large"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    if (kIndex < 0) {
        LOGGER << "Requested control point index is negative"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    int knotIndex = fSplineIndex->hostPtr()[sIndex] + 2 + kIndex;
    if (fSplineIndex->hostPtr()[sIndex+1] <= knotIndex) {
        LOGGER << "Requested control point index is two large"
                  << std::endl;
        std::runtime_error("Invalid control point being set");
    }
    fSplineKnots->hostPtr()[knotIndex] = value;
}

// A convenience function combining ReserveSpline and SetSplineKnot.
int GPUInterp::CachedWeights::AddSpline(
    int resIndex, int parIndex, double low, double high,
    double points[], int nPoints) {
    int newIndex = ReserveSpline(resIndex,parIndex,low,high,nPoints);
    int knotIndex = fSplineIndex->hostPtr()[newIndex] + 2;
    for (int p = 0; p<nPoints; ++p) {
        fSplineKnots->hostPtr()[knotIndex+p] = points[p];
    }
    return newIndex;
}

// Define CACHE_DEBUG to get lots of output from the host
#undef CACHE_DEBUG

namespace {
    // Interpolate one point.  This is the only place that changes when the
    // interpolation method changes.  This accepts a normalized "x" value, and
    // an array of control points with "dim" entries..  The control points
    // will be at (0, 1.0, 2.0, ... , dim-1).  The input variable "x" must be
    // a "floating point" index. If the index "x" is out of range, then this
    // turns into a linear extrapolation of the boundary points (try to avoid
    // that).
    //
    // Example: If the control points have dim of 5, the index "x" must be
    // greater than zero, and less than 5.  Assuming linear interpolation, an
    // input value of 2.1 results in the linear interpolation between element
    // [2] and element [3], or "(1.0-0.1)*p[2] + 0.1*p[3])".
    HEMI_DEV_CALLABLE_INLINE
    float HEMIInterp(double x, const float* points, int dim) {
#undef USE_LINEAR_INTERPOLATION
#ifdef USE_LINEAR_INTERPOLATION
        // Get the integer part and bound it to a valid index.
        int ix = x;
        if (ix < 0) ix = 0;
        if (ix > dim-2) ix = dim-2;

        // Get the remainder for the "index".  If the input index is less than
        // zero or greater than kPointSize-1, this is the distance from the
        // boundary.
        float fx = x-ix;
        float ffx = 1.0-fx;

        float v = ffx*points[ix] + fx*points[ix+1];
#endif
#define USE_SPLINE_INTERPOLATION
#ifdef USE_SPLINE_INTERPOLATION
        // Interpolate between p2 and p3
        // ix-2 ix-1 ix
        // p0   p1   p2---p3   p4   p5
        //   d10  d21  d32  d43  d54
        // m0| |m1| |m2| |m3| |m4| |m5
        //  a0 ||a1 ||a2 ||a3 ||a4 ||a5
        //     b0   b1   b2   b3   b4

        // Get the integer part
        int ix = x;

        // Calculate the indices of the two points to calculate d10
        // int d10_0 = ix-2;             // p0
        // if (d10_0<0) d10_0 = 0;
        // int d10_1 = d10_0+1;          // p1
        // Calculate the indices of the two points to calculate d21
        int d21_0 = ix-1;             // p1
        if (d21_0 < 0)     d21_0 = 0;
        if (d21_0 > dim-2) d21_0 = dim-2;
        int d21_1 = d21_0+1;          // p2
        // Calculate the indices of the two points to calculate d21
        int d32_0 = ix;               // p2
        if (d32_0 < 0)     d32_0 = 0;
        if (d32_0 > dim-2) d32_0 = dim-2;
        int d32_1 = d32_0+1;          // p3
        // Calculate the indices of the two points to calculate d43;
        int d43_0 = ix+1;             // p3
        if (d43_0 < 0)     d43_0 = 0;
        if (d43_0 > dim-2) d43_0 = dim-2;
        int d43_1 = d43_0+1;          // p4
        // Calculate the indices of the two points to calculate d43;
        int d54_0 = ix+2;             // p4
        if (d54_0 < 0)     d54_0 = 0;
        if (d54_0 > dim-2) d54_0 = dim-2;
        int d54_1 = d54_0+1;          // p5

        // Find the points to use.
        float p2 = points[d32_0];
        float p3 = points[d32_1];

        // Get the remainder for the "index".  If the input index is less than
        // zero or greater than kPointSize-1, this is the distance from the
        // boundary.
        float fx = x-d32_0;
        float fxx = fx*fx;
        float fxxx = fx*fxx;

        // Get the values of the deltas
        // float d10 = points[IX(i,d10_1)] - points[IX(i,d10_0)];
        float d21 = points[d21_1] - points[d21_0];
        float d32 = p3-p2;
        float d43 = points[d43_1] - points[d43_0];

        // Find the raw slopes at each point
        // float m1 = 0.5*(d10+d21);
        float m2 = 0.5*(d21+d32);
        float m3 = 0.5*(d32+d43);

#define USE_MONOTONIC_SPLINE
#ifdef USE_MONOTONIC_SPLINE
        float d54 = points[d54_1] - points[d54_0];
        float m4 = 0.5*(d43+d54);

        // Deal with cusp points and flat areas.
        // if (d21*d10 < 0.0) m1 = 0.0;
        if (d32*d21 <= 0.0) m2 = 0.0;
        if (d43*d32 <= 0.0) m3 = 0.0;
        if (d54*d43 <= 0.0) m4 = 0.0;

        // Find the alphas and betas
        // float a0 = (d10>0) ? m0/d21: 0;
        // float b0 = (d10>0) ? m1/d21: 0;
        // float a1 = (d21>0) ? m1/d21: 0;
        float b1 = (d21>0) ? m2/d21: 0;
        float a2 = (d32>0) ? m2/d32: 0;
        float b2 = (d32>0) ? m3/d32: 0;
        float a3 = (d43>0) ? m3/d43: 0;
        float b3 = (d43>0) ? m4/d43: 0;
        // float a4 = (d54>0) ? m4/d54: 0;
        // float b4 = (d54>0) ? m5/d54: 0;

        // Find places where can only be piecewise monotonic.
        // if (b0 <= 0) m1 = 0.0;
        if (b1 <= 0) m2 = 0.0;
        if (b2 <= 0) m3 = 0.0;
        // if (b3 <= 0) m4 = 0.0;
        // if (b4 <= 0) m5 = 0.0;
        // if (a0 <= 0) m0 = 0.0;
        // if (a1 <= 0) m1 = 0.0;
        if (a2 <= 0) m2 = 0.0;
        if (a3 <= 0) m3 = 0.0;
        // if (a4 <= 0) m4 = 0.0;

        // Limit the slopes so there isn't overshoot.  It might be
        // possible to handle the zero slope sections here without an
        // extra conditional.  if (a1 > 3 || b1 > 3) m1 = 3.0*d21;
        if (a2 > 3 || b2 > 3) m2 = 3.0*d32;
        if (a3 > 3 || b3 > 3) m3 = 3.0*d43;
        // if (a4 > 3 || b4 > 3) m4 = 3.0*d54;
#endif

        // Cubic spline with the points and slopes.
        double v = (p2*(2.0*fxxx-3.0*fxx+1.0) + m2*(fxxx-2.0*fxx+fx)
                    + p3*(3.0*fxx-2.0*fxxx) + m3*(fxxx-fxx));

#endif
        return v;
    }

#define PRINT_STEP 3

    /// Do an atomic multiplication on the GPU.  On the GPU this uses
    /// compare-and-set.  On the CPU, this is just a multiplication (no
    /// mutex, so not atomic).
    HEMI_DEV_CALLABLE_INLINE
    double HEMIAtomicMult(double* address, const double v) {
#ifndef HEMI_DEV_CODE
        // When this isn't CUDA use a simple multiplication.
        double old = *address;
        *address = *address * v;
        return old;
#else
        // When using CUDA use atomic compare-and-set to do an atomic
        // multiplication.  This only sets the result if the value at
        // address_as_ull is equal to "assumed" after the multiplication.  The
        // comparison is done with integers because of CUDA.  The return value
        // of atomicCAS is the original value at address_as_ull, so if it's
        // equal to "assumed", then nothing changed the value at the address
        // while the multiplication was being done.  If something changed the
        // value at the address "old" will not equal "assumed", and then retry
        // the multiplication.
        unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull;
        unsigned long long int assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull,
                            assumed,
                            __double_as_longlong(
                                v *  __longlong_as_double(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN
            // (since NaN != NaN)
        } while (assumed != old);
        return __longlong_as_double(old);
#endif
    }

    // A function to be used as the kernen on a CPU or GPU.  This must be
    // valid CUDA.  This sets all of the results to a fixed value.
    HEMI_KERNEL_FUNCTION(HEMISetKernel,
                         double* results,
                         const double* values,
                         const int NP) {
        for (int i : hemi::grid_stride_range(0,NP)) {
            results[i] = values[i];
#ifndef HEMI_DEV_CODE
#ifdef CACHE_DEBUG
            if (i < PRINT_STEP) {
                LOGGER << "Set kernel result " << i << " to " << results[i]
                       << std::endl;
            }
#endif
#endif
        }
    }

    // A function to be used as the kernen on a CPU or GPU.  This must be
    // valid CUDA.  This accumulates the normalization parameters into the
    // results.
    HEMI_KERNEL_FUNCTION(HEMINormsKernel,
                         double* results,
                         const double* params,
                         const int* rIndex,
                         const short* pIndex,
                         const int NP) {
        for (int i : hemi::grid_stride_range(0,NP)) {
            HEMIAtomicMult(&results[rIndex[i]], params[pIndex[i]]);
#ifndef HEMI_DEV_CODE
#ifdef CACHE_DEBUG
            if (rIndex[i] < PRINT_STEP) {
                LOGGER << "Norms kernel " << i
                       << " iEvt " << rIndex[i]
                       << " iPar " << pIndex[i]
                       << " = " << params[pIndex[i]]
                       << std::endl;
            }
#endif
#endif
        }
    }

    // A function to be used as the kernel on either the CPU or GPU.  This
    // must be valid CUDA coda.
    HEMI_KERNEL_FUNCTION(HEMISplinesKernel,
                         double* results,
                         const double* params,
                         const float* lowerClamp,
                         const float* upperClamp,
                         const float* knots,
                         const int* rIndex,
                         const short* pIndex,
                         const int* sIndex,
                         const int NP) {
        for (int i : hemi::grid_stride_range(0,NP)) {
            const int id0 = sIndex[i];
            const int id1 = sIndex[i+1];
            double x = params[pIndex[i]];
            x = (x-knots[id0])*knots[id0+1];
            x = HEMIInterp(x, &knots[id0+2], id1-id0-2);
            float lc = lowerClamp[pIndex[i]];
            if (x < lc) x = lc;
            float uc = upperClamp[pIndex[i]];
            if (x > uc) x = uc;
            HEMIAtomicMult(&results[rIndex[i]], x);
#ifndef HEMI_DEV_CODE
#ifdef CACHE_DEBUG
            if (rIndex[i] < PRINT_STEP) {
                LOGGER << "Splines kernel " << i
                       << " iEvt " << rIndex[i]
                       << " iPar " << pIndex[i]
                       << " = " << params[pIndex[i]]
                       << " --> " << x
                       << std::endl;
            }
#endif
#endif
        }
    }
}

void GPUInterp::CachedWeights::UpdateResults() {
    for (int i = 0; i<GetParameterCount(); ++i) {
        double lm = fLowerMirror->at(i);
        double um = fUpperMirror->at(i);
        double v = GetParameter(i);
        // Mirror the input value at lm and um.
        int brake = 20;
        while (v < lm || v > um) {
            if (v < lm) v = lm + (lm - v);
            if (v > um) v = um - (v - um);
            if (--brake < 1) throw;
        }
        SetParameter(i,v);
    }

    HEMISetKernel setKernel;
#ifdef FORCE_HOST_KERNEL
    setKernel(   fResults->hostPtr(),
                 fInitialValues->hostPtr(),
                 GetResultCount());
#else
    hemi::launch(setKernel,
                 fResults->writeOnlyPtr(),
                 fInitialValues->readOnlyPtr(),
                 GetResultCount());
#endif

    HEMINormsKernel normsKernel;
#ifdef FORCE_HOST_KERNEL
    normsKernel(fResults->hostPtr(),
                fParameters->hostPtr(),
                fNormResult->hostPtr(),
                fNormParameter->hostPtr(),
                GetNormsUsed());
#else
    hemi::launch(normsKernel,
                 fResults->writeOnlyPtr(),
                 fParameters->readOnlyPtr(),
                 fNormResult->readOnlyPtr(),
                 fNormParameter->readOnlyPtr(),
                 GetNormsUsed());
#endif

    HEMISplinesKernel splinesKernel;
#ifdef FORCE_HOST_KERNEL
    splinesKernel(     fResults->hostPtr(),
                       fParameters->hostPtr(),
                       fLowerClamp->hostPtr(),
                       fUpperClamp->hostPtr(),
                       fSplineKnots->hostPtr(),
                       fSplineResult->hostPtr(),
                       fSplineParameter->hostPtr(),
                       fSplineIndex->hostPtr(),
                       GetSplinesUsed()
        );
#else
    hemi::launch(splinesKernel,
                 fResults->writeOnlyPtr(),
                 fParameters->readOnlyPtr(),
                 fLowerClamp->readOnlyPtr(),
                 fUpperClamp->readOnlyPtr(),
                 fSplineKnots->readOnlyPtr(),
                 fSplineResult->readOnlyPtr(),
                 fSplineParameter->readOnlyPtr(),
                 fSplineIndex->readOnlyPtr(),
                 GetSplinesUsed()
        );
#endif

    fResults->hostPtr();  // A simple way to copy from the device
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
