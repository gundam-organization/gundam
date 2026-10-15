#include <iostream>
#include <limits>
#include <vector>

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

    // Some histogram bins do not receive any events.  The sum (and sum
    // squared) for an empty bin must be zero.  The first bin, a bin in the
    // middle, and the last bin are left empty.  The empty last bin checks
    // that the result is not read from past the end of the internal work
    // buffer (where fBinOffsets[bin] == fBinOffsets[bin+1] == N).
    std::vector<int> emptyBins{0, 3, bins-1};
    std::vector<bool> isEmpty(bins, false);
    for (int b : emptyBins) isEmpty[b] = true;

    int filledBins = bins - int(emptyBins.size());
    int N = filledBins*entries;

    // Give each bin a distinct weight so that a sum read from the wrong bin
    // is unambiguous.
    hemi::Array<double> weights(N);
    std::vector<double> expectedSum(bins, 0.0);
    std::vector<double> expectedSum2(bins, 0.0);

    Cache::RecursiveSums recursiveSums(weights,bins);
    {
        int e = 0;
        for (int b = 0; b < bins; ++b) {
            if (isEmpty[b]) continue;
            double w = 1.0 + b;
            expectedSum[b] = entries*w;
            expectedSum2[b] = entries*w*w;
            for (int i = 0; i < entries; ++i, ++e) {
                weights.hostPtr()[e] = w;
                recursiveSums.SetEventIndex(e,b);
            }
        }
        ASSERT_EQ(e, N);
    }

    recursiveSums.Initialize();

    for (int i=0; i<100; ++i) {
        recursiveSums.Reset();
        recursiveSums.Apply();
    }

    hemi::deviceSynchronize();

    // Every bin must be correct.  The filled bins must hold the sum of
    // their weights, and the empty bins (fBinOffsets[bin] ==
    // fBinOffsets[bin+1]) must be zero rather than the sum of the next
    // filled bin.
    for (int b = 0; b < bins; ++b) {
        const char* kind = isEmpty[b] ? "empty bin " : "bin ";
        EXPECT_DOUBLE_EQ(recursiveSums.GetSum(b), expectedSum[b])
            << "RecursiveSums sum in " << kind << b << " is wrong";
        EXPECT_DOUBLE_EQ(recursiveSums.GetSum2(b), expectedSum2[b])
            << "RecursiveSums sum2 in " << kind << b << " is wrong";
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
