#ifndef GUNDAM_CUDA_BINNING_H
#define GUNDAM_CUDA_BINNING_H

#include "CudaWeightModel.h"
#include "Semantics/BackendSemanticsQualifiers.h"

#include <cmath>
#include <limits>

namespace Backends {
  struct CudaBinContribution {
    std::size_t event{0}; // Result slot, independent of event traversal order.
    int bin{-1}; // Global histogram bin, including sample outside bins.
    double fraction{0};
  };

  struct CudaBinningResult {
    std::size_t count{0}, largestBin{0};
    double total{0}, excess{0};
    bool isValid{true};
  };

  inline constexpr double cudaBinningEpsilon = std::numeric_limits<double>::epsilon();

  namespace Semantics {
    GUNDAM_BACKEND_HOST_DEVICE inline bool cudaBinningIsFinite(double value_) {
#ifdef __CUDA_ARCH__
      return ::isfinite(value_);
#else
      return std::isfinite(value_);
#endif
    }

    GUNDAM_BACKEND_HOST_DEVICE inline double cudaBinFraction(
        const BinningBin& bin_, const BinningInterval* intervals_, const double* values_) {
      double fraction = 1.;
      for( std::size_t j = bin_.firstInterval; j < bin_.firstInterval + bin_.intervalCount; ++j ){
        const auto& interval = intervals_[j];
        fraction *= GundamBinningMath::intervalFraction(interval.lowerBound, interval.upperBound,
            interval.lowerInverseWidth, interval.upperInverseWidth, values_[interval.variableIndex]);
        if( fraction == 0. ){ break; }
      }
      return fraction;
    }

    // Same event-level normalization rule as CPU: tolerate only roundoff above
    // one, subtract that excess from the first largest contribution, and retain
    // the outside remainder. Analysis and writing share the fraction evaluator.
    GUNDAM_BACKEND_HOST_DEVICE inline CudaBinningResult analyzeCudaBinning(
        const CudaBinningSample& sample_, const BinningBin* bins_, const BinningInterval* intervals_, const double* values_) {
      CudaBinningResult result;
      for( std::size_t j = 0; j < sample_.variableCount; ++j ){
        if( not cudaBinningIsFinite(values_[j]) ){ result.isValid = false; return result; }
      }
      double largest = -1.;
      for( std::size_t j = sample_.firstBin; j < sample_.endBin; ++j ){
        const double fraction = cudaBinFraction(bins_[j], intervals_, values_);
        if( not cudaBinningIsFinite(fraction) or fraction < 0. ){ result.isValid = false; return result; }
        if( fraction == 0. ){ continue; }
        ++result.count;
        result.total += fraction;
        if( fraction > largest ){ largest = fraction; result.largestBin = j; }
      }
      const double tolerance = 64. * cudaBinningEpsilon * double(result.count ? result.count : 1);
      if( not cudaBinningIsFinite(result.total) or result.total > 1. + tolerance ){
        result.isValid = false;
        return result;
      }
      if( result.total > 1. ){
        result.excess = result.total - 1.;
        result.total = 1.;
      }
      if( result.total < 1. ){ ++result.count; }
      return result;
    }

    GUNDAM_BACKEND_HOST_DEVICE inline bool writeCudaBinning(
        const CudaBinningSample& sample_, const BinningBin* bins_, const BinningInterval* intervals_, const double* values_,
        const CudaBinningResult& result_, std::size_t event_, int outsideBin_, CudaBinContribution* output_, std::size_t capacity_) {
      if( not result_.isValid ){ return false; }
      std::size_t count = 0;
      for( std::size_t j = sample_.firstBin; j < sample_.endBin; ++j ){
        double fraction = cudaBinFraction(bins_[j], intervals_, values_);
        if( fraction == 0. ){ continue; }
        if( j == result_.largestBin ){ fraction -= result_.excess; }
        if( count >= capacity_ ){ return false; }
        output_[count++] = {event_, sample_.binOffset + bins_[j].binIndex, fraction};
      }
      if( result_.total < 1. ){
        if( count >= capacity_ ){ return false; }
        output_[count++] = {event_, outsideBin_, 1. - result_.total};
      }
      const bool complete = count == capacity_;
      // Keep the GPU reduction safe even if a count/write disagreement is
      // detected. The request is rejected before publishing any result.
      while( count < capacity_ ){ output_[count++] = {event_, outsideBin_, 0.}; }
      return complete;
    }
  }
}
#endif
