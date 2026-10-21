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

}

#endif // GUNDAM_CPU_WEIGHT_KERNELS_H
