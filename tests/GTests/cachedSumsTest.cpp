#include <iostream>
#include <limits>

#include <TRandom.h>

#include "Logger.h"

#include "CacheIndexedSums.h"
#include "CacheRecursiveSums.h"

#include "gtest/gtest.h"

#ifdef HEMI_CUDA_COMPILER
#define ASSERT_SUCCESS(res) ASSERT_EQ(cudaSuccess, (res));
#define ASSERT_FAILURE(res) ASSERT_NE(cudaSuccess, (res));
#define cachedSumsTest cachedSumsTestDevice
#else
#define ASSERT_SUCCESS(res)
#define ASSERT_FAILURE(res)
#define cachedSumsTest cachedSumsTestHost
#endif

TEST(cachedSumsTest, IndexedSums)
{
    int entries = 100;
    int bins = 10;
    int N = bins*entries;
    hemi::Array<double> weights(N);
    for (int e=0; e<N; ++e) {
        weights.hostPtr()[e] = 1.0;
    }

    Cache::IndexedSums indexedSums(weights,bins);
    for (int e=0; e<N; ++e) {
        int bin = e/entries;
        indexedSums.SetEventIndex(e,bin);
    }

    indexedSums.Initialize();

    for (int i=0; i<100; ++i) {
        indexedSums.Reset();
        indexedSums.Apply();
    }

    hemi::deviceSynchronize();

    for (int b = 0; b < bins; ++b) {
        EXPECT_EQ(indexedSums.GetSum(b), entries)
            << "IndexedSums bin " << b << " is wrong";
    }

}

TEST(cachedSumsTest, RecursiveSums)
{
    int entries = 100;
    int bins = 10;
    int N = bins*entries;
    hemi::Array<double> weights(N);
    for (int e=0; e<N; ++e) {
        weights.hostPtr()[e] = 1.0;
    }

    Cache::RecursiveSums recursiveSums(weights,bins);
    for (int e=0; e<N; ++e) {
        int bin = e/entries;
        recursiveSums.SetEventIndex(e,bin);
    }

    recursiveSums.Initialize();

    for (int i=0; i<100; ++i) {
        recursiveSums.Reset();
        recursiveSums.Apply();
    }

    hemi::deviceSynchronize();

    for (int b = 0; b < bins; ++b) {
        EXPECT_EQ(recursiveSums.GetSum(b), entries)
            << "RecursiveSums bin " << b << " is wrong";
    }

}

TEST(cachedSumsTest, CompareSums)
{
    int entries = 100;
    int bins = 10;
    int N = bins*entries;
    hemi::Array<double> weights(N);
    for (int e=0; e<N; ++e) {
        weights.hostPtr()[e] = 1.0;
    }

    gRandom->SetSeed(0);

    Cache::IndexedSums indexedSums(weights,bins);
    Cache::RecursiveSums recursiveSums(weights,bins);

    for (int e=0; e<N; ++e) {
        int bin = e/entries;
        indexedSums.SetEventIndex(e,bin);
        recursiveSums.SetEventIndex(e,bin);
    }

    indexedSums.Initialize();
    recursiveSums.Initialize();

    for (int i=0; i<100; ++i) {
        for (int e=0; e<N; ++e) {
            weights.hostPtr()[e] = gRandom->Gaus(1.0,0.1);
        }
        indexedSums.Reset();
        indexedSums.Apply();
        recursiveSums.Reset();
        recursiveSums.Apply();
        for (int b = 0; b < bins; ++b) {
            // Estimate how close the two results should be.  This needs
            // to allow for numeric precision, and the magnitude of the
            // values.  Most bugs will be "large" mistakes.
            double err = 100*std::numeric_limits<double>::epsilon()
                *(indexedSums.GetSum(b)+recursiveSums.GetSum(b));
            EXPECT_NEAR(indexedSums.GetSum(b), recursiveSums.GetSum(b), err)
                << "Mismatch between indexed and recursive sum in bin " << b;
        }
    }

}
