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

  template<typename ResponseEvaluator>
  inline double evalVariableResponse(const BackendVariableResponseDescriptor& target_,
                                     const std::uint32_t* dialIndices_, ResponseEvaluator&& response_) {
    double value = target_.overrideDial == std::size_t(-1) ? target_.referenceValue : response_(target_.overrideDial);
    for( std::size_t i = target_.firstMultiplyDial; i < target_.firstMultiplyDial + target_.multiplyDialCount; ++i ){
      value *= response_(dialIndices_[i]);
    }
    for( std::size_t i = target_.firstAddDial; i < target_.firstAddDial + target_.addDialCount; ++i ){
      value += response_(dialIndices_[i]);
    }
    return value;
  }

  inline double evalEventWeight(const PropagationView& propagation_,
                                const EventView& eventRef_,
                                const PropagationInputs& inputs_) {
    double weight = 1.;
    for( std::size_t iDial = 0; iDial < eventRef_.weight.dialCount; ++iDial ){
      const auto& dial = propagation_.dials[propagation_.eventDialIndices[eventRef_.weight.firstDial + iDial]];
      weight *= evalDialResponse(propagation_, dial, inputs_);
    }
    if( propagation_.hasReweightCap and weight > propagation_.maxReweight ){ weight = propagation_.maxReweight; }
    return eventRef_.weight.baseWeight * weight;
  }

}

#endif // GUNDAM_BACKEND_DIAL_SEMANTICS_H
