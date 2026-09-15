#ifndef GUNDAM_BACKEND_DIAL_SEMANTICS_H
#define GUNDAM_BACKEND_DIAL_SEMANTICS_H

#include "EngineView.h"
#include "PropagationInputs.h"
#include "Semantics/BackendDialSemanticsCore.h"

namespace Backends::Semantics {
  inline const double* getParameterValues(const PropagationInputs& inputs_) {
    return inputs_.parameters.values.data();
  }

  inline double evalDialResponse(const PropagationView& propagation_,
                                 const BackendDialDescriptor& dialRef_,
                                 const PropagationInputs& inputs_) {
    if( dialRef_.type == BackendDialType::ExternalWeight ){
      const auto& block = inputs_.externalWeights[dialRef_.externalBlockIndex];
      return clampDialResponse(dialRef_, block.values[dialRef_.externalWeightIndex]);
    }
    return evalDialResponse(
        dialRef_,
        propagation_.dialInputs.data(),
        propagation_.dialPayloads.data(),
        getParameterValues(inputs_)
    );
  }

  inline double evalEventWeight(const PropagationView& propagation_,
                                const EventView& eventRef_,
                                const PropagationInputs& inputs_) {
    double weight = eventRef_.weight.baseWeight;
    for( std::size_t iDial = 0; iDial < eventRef_.weight.dialCount; ++iDial ){
      const auto& dial = propagation_.dials[propagation_.eventDialIndices[eventRef_.weight.firstDial + iDial]];
      weight *= evalDialResponse(propagation_, dial, inputs_);
    }
    return weight;
  }

}

#endif // GUNDAM_BACKEND_DIAL_SEMANTICS_H
