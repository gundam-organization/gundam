#ifndef GUNDAM_CPU_WEIGHT_KERNELS_H
#define GUNDAM_CPU_WEIGHT_KERNELS_H

#include "EngineDescriptors.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace Backends::CpuWeightKernels {

  // Independent accumulators overlap response loads without changing the
  // multiplication order within any event. Only cache reads are interleaved;
  // mixed evaluators retain their original event-by-event call order.
  template<std::size_t lanes, bool useReweightCap>
  void reweightInterleaved(const BackendEventWeightDescriptor* events_,
                          const std::uint32_t* dialIndices_, const double* responses_,
                          double* weights_, std::size_t begin_, std::size_t end_,
                          double maxReweight_) {
    static_assert(lanes == 2 or lanes == 4, "Supported interleave widths are 2 and 4.");
    std::size_t event = begin_;
    for( ; end_ - event >= lanes; event += lanes ){
      std::array<double, lanes> products;
      products.fill(1.);
      std::size_t longest{0};
      for( std::size_t lane = 0; lane < lanes; ++lane ){
        longest = std::max(longest, events_[event + lane].dialCount);
      }
      for( std::size_t dial = 0; dial < longest; ++dial ){
        for( std::size_t lane = 0; lane < lanes; ++lane ){
          const auto& descriptor = events_[event + lane];
          if( dial < descriptor.dialCount ){
            products[lane] *= responses_[dialIndices_[descriptor.firstDial + dial]];
          }
        }
      }
      for( std::size_t lane = 0; lane < lanes; ++lane ){
        if constexpr( useReweightCap ){ products[lane] = std::min(products[lane], maxReweight_); }
        weights_[event + lane] = events_[event + lane].baseWeight * products[lane];
      }
    }
    // Short ranges and the remaining one to three events use the scalar order.
    for( ; event < end_; ++event ){
      const auto& descriptor = events_[event];
      double product = 1.;
      for( std::size_t dial = 0; dial < descriptor.dialCount; ++dial ){
        product *= responses_[dialIndices_[descriptor.firstDial + dial]];
      }
      if constexpr( useReweightCap ){ product = std::min(product, maxReweight_); }
      weights_[event] = descriptor.baseWeight * product;
    }
  }

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

}

#endif // GUNDAM_CPU_WEIGHT_KERNELS_H
