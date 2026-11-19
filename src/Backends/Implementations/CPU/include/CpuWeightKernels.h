#ifndef GUNDAM_CPU_WEIGHT_KERNELS_H
#define GUNDAM_CPU_WEIGHT_KERNELS_H

#include "EngineDescriptors.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace Backends::CpuWeightKernels {

  // Both execution paths consume the same resident buffers. Preserve the dial
  // product order and apply the cap before multiplying by the base weight.
  template<bool allResponsesCached, bool useReweightCap, typename ResponseEvaluator>
  void reweight(const BackendEventWeightDescriptor* events_,
                const std::uint32_t* dialIndices_, const double* responses_,
                double* weights_, std::size_t begin_, std::size_t end_,
                double maxReweight_, ResponseEvaluator&& evaluate_) {
    for( std::size_t iEvent = begin_; iEvent < end_; ++iEvent ){
      const auto& event = events_[iEvent];
      double weight = 1.;
      for( std::size_t iDial = 0; iDial < event.dialCount; ++iDial ){
        const auto dial = dialIndices_[event.firstDial + iDial];
        if constexpr( allResponsesCached ){ weight *= responses_[dial]; }
        else{ weight *= evaluate_(dial); }
      }
      if constexpr( useReweightCap ){ weight = std::min(weight, maxReweight_); }
      weights_[iEvent] = event.baseWeight * weight;
    }
  }

  // Independent products can overlap their response loads without changing
  // the multiplication order within either event.
  template<bool allResponsesCached, bool useReweightCap, typename ResponseEvaluator>
  void reweightInterleaved(const BackendEventWeightDescriptor* events_,
                          const std::uint32_t* dialIndices_, const double* responses_,
                          double* weights_, std::size_t begin_, std::size_t end_,
                          double maxReweight_, ResponseEvaluator&& evaluate_) {
    if constexpr( not allResponsesCached ){
      reweight<allResponsesCached, useReweightCap>(events_, dialIndices_, responses_, weights_,
                                                 begin_, end_, maxReweight_, evaluate_);
      return;
    }
    std::size_t iEvent = begin_;
    for( ; end_ - iEvent >= 2; iEvent += 2 ){
      const auto& first = events_[iEvent];
      const auto& second = events_[iEvent + 1];
      const auto* firstDials = dialIndices_ + first.firstDial;
      const auto* secondDials = dialIndices_ + second.firstDial;
      const auto common = std::min(first.dialCount, second.dialCount);
      double firstWeight{1.}, secondWeight{1.};
      for( std::size_t i = 0; i < common; ++i ){
        firstWeight *= responses_[firstDials[i]];
        secondWeight *= responses_[secondDials[i]];
      }
      for( std::size_t i = common; i < first.dialCount; ++i ){ firstWeight *= responses_[firstDials[i]]; }
      for( std::size_t i = common; i < second.dialCount; ++i ){ secondWeight *= responses_[secondDials[i]]; }
      if constexpr( useReweightCap ){
        firstWeight = std::min(firstWeight, maxReweight_);
        secondWeight = std::min(secondWeight, maxReweight_);
      }
      weights_[iEvent] = first.baseWeight * firstWeight;
      weights_[iEvent + 1] = second.baseWeight * secondWeight;
    }
    reweight<allResponsesCached, useReweightCap>(events_, dialIndices_, responses_, weights_,
                                               iEvent, end_, maxReweight_, evaluate_);
  }

}

#endif // GUNDAM_CPU_WEIGHT_KERNELS_H
