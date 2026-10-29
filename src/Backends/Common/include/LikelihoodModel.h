#ifndef GUNDAM_LIKELIHOOD_MODEL_H
#define GUNDAM_LIKELIHOOD_MODEL_H
#include "EngineView.h"
#include "DeviceMath/DoubleFloat.h"
#include <type_traits>
#include <limits>
#include <stdexcept>

namespace Backends {
  inline bool hasPortableLikelihood(const LikelihoodView& view_) {
    if( view_.empty() ){ return false; }
    for( const auto& sample : view_.samples ){
      if( sample.portableOptions.kind == LikelihoodKind::Host ){ return false; }
    }
    return true;
  }
  template<typename Scalar> Scalar likelihoodScalar(double value_, bool allowLargeLimit_ = false) {
    if constexpr( std::is_same_v<Scalar, GundamDeviceMath::DoubleFloat> ){
      if( allowLargeLimit_ && value_ > std::numeric_limits<float>::max() ){
        return Scalar(std::numeric_limits<float>::infinity());
      }
      if( !std::isfinite(value_) || std::abs(value_) > std::numeric_limits<float>::max()
          || (value_ != 0. && std::abs(value_) < std::numeric_limits<float>::min()) ){
        throw std::runtime_error("Likelihood value is outside the Metal scalar exponent range.");
      }
    }
    return Scalar(value_);
  }
  template<typename Scalar> std::vector<LikelihoodBin<Scalar>> packLikelihood(const LikelihoodView& view_, int totalBins_) {
    std::vector<LikelihoodBin<Scalar>> bins;
    if( !hasPortableLikelihood(view_) ){ return bins; }
    for( const auto& sample : view_.samples ){
      if( sample.binOffset < 0 || std::size_t(sample.binOffset) + sample.dataSums.size() > std::size_t(totalBins_) ){
        throw std::runtime_error("Likelihood bin range exceeds model histograms.");
      }
      if( sample.portableOptions.useNominalVariance && sample.nominalVariances.size() != sample.dataSums.size() ){
        throw std::runtime_error("Likelihood nominal variance count mismatch.");
      }
      const auto& o = sample.portableOptions;
      for( std::size_t i = 0; i < sample.dataSums.size(); ++i ){
        LikelihoodBin<Scalar> bin;
        bin.globalBin = sample.binOffset + i;
        bin.ignored = i < sample.ignoredBins.size() && sample.ignoredBins[i];
        bin.data = likelihoodScalar<Scalar>(sample.dataSums[i]);
        bin.nominalVariance = o.useNominalVariance ? likelihoodScalar<Scalar>(sample.nominalVariances[i]) : Scalar(0);
        bin.options = {o.kind, o.usePoissonLikelihood, o.useNominalVariance, o.throwIfInfLlh,
                       likelihoodScalar<Scalar>(o.fractionalErrorLimit, true), likelihoodScalar<Scalar>(o.expectedValueMinimum)};
        bins.push_back(bin);
      }
    }
    return bins;
  }
}
#endif
