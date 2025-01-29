#ifndef GundamAlmostEqual_h_seen
#define GundamAlmostEqual_h_seen
#include <iostream>
#include <cmath>
#include <limits>

namespace GundamUtils {

    /// Check if two real numbers are close enough to be considered equal.
    /// This takes into account different effects: 1) precision of the
    /// underlying numerical representation; 2) the magnitudes of the numbers
    /// being compared; and 3), the fraction of bits that are expected to have
    /// been lost during the numeric calculation.  The fractions of bits lost
    /// will be a number between 0.0 and 1.0, where 0 means no bits are lost
    /// due to numerical problems, and 1 means all bits are lost.  The number
    /// of bits grows (very) roughly like the square root of the numbers of
    /// operations, but depends heavily on the actual sequence of operations.
    /// Losing 30% of the accuracy is a plausible default bound.
    ///
    /// Note: This is intended for testing and debugging.  It is not intended
    /// for "production", and is written to be painfully clear about it's
    /// assumptions.
    template <typename U, typename V>
    bool almostEqual(U A, V B, double fractionOfBitsLost = 0.3) {
        // Non-finite numbers are never equal.
        if (not std::isfinite(A) || not std::isfinite(B)) return false;

        // Internally, operate on actual doubles.  This is for simplicity and
        // clarity.
        const double a = A;
        const double b = B;

        // Find the precision of the least precise value.
        double epsA = std::numeric_limits<U>::epsilon();
        double epsB = std::numeric_limits<V>::epsilon();
        double eps = std::max(epsA, epsB);  // Take the worst precision

        // Reduce the precision:  This accounts for the fraction of bits of
        // accuracy that has been lost during the calculation.
        if (fractionOfBitsLost > 0.0) {
            fractionOfBitsLost = std::min(fractionOfBitsLost, 0.99);
            eps = std::pow(eps, (1.0-fractionOfBitsLost));
        }

        // Scale to the value magnitudes.  Don't let the precision become
        // subnormal.  The standard binary representation for a floating point
        // number intends to have the first bit be "1" (called normal).  A
        // subnormal representation means that the first bit is zero.  For
        // instance (using decimal), 0.00123 has a normal representatin of
        // 1.23E-3, and (one possible) subnormal representation of 0.0123E-1
        // (you can find references on the web that are more precise).
        if (std::isnormal(a*eps)) epsA = std::abs(a*eps);
        if (std::isnormal(b*eps)) epsB = std::abs(b*eps);
        double finalEPS = std::max(epsA, epsB);

        // Check that the difference is less than the acceptable precision
        double diff = std::abs(a - b);
        return (diff < finalEPS);
    }
}
#endif
