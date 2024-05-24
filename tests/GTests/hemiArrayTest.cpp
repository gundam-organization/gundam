#include <iostream>
#include "hemi/array.h"
#include "hemi/launch.h"
#include "hemi/grid_stride_range.h"
#include "CacheAtomicAdd.h"
#include "CacheAtomicSet.h"
#include <algorithm>

#include "gtest/gtest.h"

#ifdef HEMI_CUDA_COMPILER
#define ASSERT_SUCCESS(res) ASSERT_EQ(cudaSuccess, (res));
#define ASSERT_FAILURE(res) ASSERT_NE(cudaSuccess, (res));
#define hemiArrayTest hemiArrayTestDevice
#else
#define ASSERT_SUCCESS(res)
#define ASSERT_FAILURE(res)
#define hemiArrayTest hemiArrayTestHost
#endif

TEST(hemiArrayTest, CreatesAndFillsArrayOnHost)
{
    const int n = 1000000;
    const double val = 3.14159;
    hemi::Array<double> data(n);

    ASSERT_EQ(data.size(), n);

    double *ptr = data.writeOnlyHostPtr();
    std::fill(ptr, ptr+n, val);

    for(int i = 0; i < n; i++) {
        EXPECT_EQ(val, data.readOnlyHostPtr()[i]);
    }
}

namespace {
    // A function to be used as the kernel on either the CPU or GPU.  This
    // must be valid CUDA coda.
    HEMI_KERNEL_FUNCTION(HEMIFill, double* ptr, int N, double val) {
        for (int i : hemi::grid_stride_range(0,N)) {
            ptr[i] = val;
        }
    }
}

TEST(hemiArrayTest, CreatesAndFillsArrayOnDevice)
{
    const int n = 1000000;
    const double val = 3.14159;
    hemi::Array<double> data(n);

    HEMIFill fillArray;
    hemi::launch(fillArray,data.writeOnlyPtr(),n,val);

    for(int i = 0; i < n; i++) {
        EXPECT_EQ(val, data.readOnlyPtr(hemi::host)[i])
            << "Array value mismatched";
    }

    ASSERT_SUCCESS(hemi::deviceSynchronize());
}

namespace {
    // A function to be used as the kernel on either the CPU or GPU.  This
    // must be valid CUDA coda.
    HEMI_KERNEL_FUNCTION(HEMISquare, double* ptr, int N) {
        for (int i : hemi::grid_stride_range(0,N)) {
            double value = ptr[i]*ptr[i];
            CacheAtomicSet(&ptr[i],value);
        }
    }
}

TEST(hemiArrayTest, FillsOnHostModifiesOnDevice)
{
    const int n = 100;
    double val = 2.0;
    hemi::Array<double> data(n);

    ASSERT_EQ(data.size(), n);

    double *ptr = data.writeOnlyHostPtr();
    std::fill(ptr, ptr+n, val);

    HEMISquare squareArray;
    hemi::launch(squareArray,data.ptr(),data.size()); val *= val;
    hemi::launch(squareArray,data.writeOnlyPtr(),data.size()); val *= val;
    hemi::launch(squareArray,data.writeOnlyPtr(),data.size()); val *= val;
    hemi::launch(squareArray,data.writeOnlyPtr(),data.size()); val *= val;

    for(int i = 0; i < n; i++) {
        double result = data.readOnlyPtr(hemi::host)[i];
        EXPECT_EQ(val,result)
            << "Mismatch at element " << i
            << " current: " << data.readOnlyPtr(hemi::host)[i];
    }

    ASSERT_SUCCESS(hemi::deviceSynchronize());
}

namespace {
    // A function to be used as the kernel on either the CPU or GPU.  This
    // must be valid CUDA coda.
    HEMI_KERNEL_FUNCTION(HEMICoverage, double* ptr, int N) {
        for (int i : hemi::grid_stride_range(0,N)) {
            CacheAtomicAdd(&ptr[i],1.0*i+1.0);
        }
    }
}

TEST(hemiArrayTest, CheckCoverageInKernel)
{
    const int n = 100;
    hemi::Array<double> data(n);

    ASSERT_EQ(data.size(), n);

    double *ptr = data.writeOnlyHostPtr();
    std::fill(ptr, ptr+n, 0.0);

    HEMICoverage kernelCoverage;
    hemi::launch(kernelCoverage,data.ptr(),data.size());
    hemi::launch(kernelCoverage,data.writeOnlyPtr(),data.size());
    hemi::launch(kernelCoverage,data.writeOnlyPtr(),data.size());

    for(int i = 0; i < n; i++) {
        EXPECT_EQ(3.0*(i+1), data.readOnlyPtr(hemi::host)[i])
            << "Elements not covered by kernel";
    }

    ASSERT_SUCCESS(hemi::deviceSynchronize());
}

namespace {
    // A function to be used as the kernel on either the CPU or GPU.  This
    // must be valid CUDA coda.
    HEMI_KERNEL_FUNCTION(HEMIAccumulate, double* result, int N) {
        for (int i : hemi::grid_stride_range(0,N)) {
            CacheAtomicAdd(result,1.0);
        }
    }
}

TEST(hemiArrayTest, AccumulatesOnDevice)
{
    hemi::Array<double> data(1);

    ASSERT_EQ(data.size(), 1);

    double *ptr = data.writeOnlyHostPtr();
    std::fill(ptr, ptr+1, 0.0);

    HEMIAccumulate arrayAccumulate;
    const int maximum = 1000;
    hemi::launch(arrayAccumulate,data.ptr(),maximum);

    ASSERT_EQ(maximum, data.readOnlyPtr(hemi::host)[0]);

    ASSERT_SUCCESS(hemi::deviceSynchronize());
}
