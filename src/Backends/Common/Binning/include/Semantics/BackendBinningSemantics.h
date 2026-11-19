#ifndef GUNDAM_BACKEND_BINNING_SEMANTICS_H
#define GUNDAM_BACKEND_BINNING_SEMANTICS_H

#ifndef __METAL_VERSION__
#include "DeviceMath/SharedBinningMath.h"
#endif

#ifdef __CUDACC__
#define GUNDAM_BINNING_CALLABLE __host__ __device__
#else
#define GUNDAM_BINNING_CALLABLE
#endif

namespace Backends::BinningSemantics {
  // Pointer types are deduced, including Metal device address spaces. Search
  // order and the fused strict/smoothed Metal traversal belong to the caller.
  template<typename Bin, typename Intervals, typename Values>
  GUNDAM_BINNING_CALLABLE inline bool containsStrictBin(Bin bin_, Intervals intervals_, Values values_) {
    for( auto i = bin_.firstInterval; i < bin_.firstInterval + bin_.intervalCount; ++i ){
      const auto& interval = intervals_[i];
      if( !GundamBinningMath::contains(interval.lowerBound, interval.upperBound, values_[interval.variableIndex]) ){
        return false;
      }
    }
    return true;
  }

  template<typename Bin, typename Intervals, typename Values>
  GUNDAM_BINNING_CALLABLE inline auto binFraction(Bin bin_, Intervals intervals_, Values values_)
      -> decltype(intervals_[0].lowerBound) {
    using Scalar = decltype(intervals_[0].lowerBound);
    Scalar fraction(1);
    for( auto i = bin_.firstInterval; i < bin_.firstInterval + bin_.intervalCount; ++i ){
      const auto& interval = intervals_[i];
      fraction *= GundamBinningMath::intervalFraction(interval.lowerBound, interval.upperBound,
          interval.lowerInverseWidth, interval.upperInverseWidth, values_[interval.variableIndex]);
      if( fraction == 0 ){ break; }
    }
    return fraction;
  }

  template<typename Scalar> struct BinningTotal {
    Scalar total, excess;
    bool isValid;
  };

  // The caller supplies its existing tolerance and finiteness check. Keep
  // largest-contribution selection, correction and outside storage local.
  template<typename Scalar, typename Tolerance>
  GUNDAM_BINNING_CALLABLE inline BinningTotal<Scalar> finalizeBinningTotal(
      Scalar total_, Tolerance tolerance_, bool isFinite_) {
    BinningTotal<Scalar> result{total_, Scalar(0), isFinite_ && !(total_ > 1 + tolerance_)};
    if( result.isValid && total_ > 1 ){
      result.excess = total_ - 1;
      result.total = 1;
    }
    return result;
  }
}
#undef GUNDAM_BINNING_CALLABLE
#endif
