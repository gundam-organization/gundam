#ifndef GUNDAM_BACKEND_DIAL_SEMANTICS_H
#define GUNDAM_BACKEND_DIAL_SEMANTICS_H

#include "EngineView.h"
#include "PropagationInputs.h"
#include "Semantics/BackendDialSemanticsCore.h"

#include <cmath>
#include <stdexcept>

namespace Backends::Semantics {
  // Validate once while preparing the model, never in the evaluation loop.
  inline void validateSurfaceDial(const PropagationView& model_, const BackendDialDescriptor& dial_) {
    if( dial_.type != BackendDialType::Bilinear and dial_.type != BackendDialType::Bicubic ){ return; }
    if( dial_.inputCount != 2 or dial_.firstInput > model_.dialInputs.size()
        or dial_.inputCount > model_.dialInputs.size() - dial_.firstInput ){
      throw std::runtime_error("Surface dials require exactly two valid inputs.");
    }
    if( dial_.payloadOffset > model_.dialPayloads.size()
        or dial_.payloadSize > model_.dialPayloads.size() - dial_.payloadOffset or dial_.payloadSize < 10 ){
      throw std::runtime_error("Surface dial payload is incomplete.");
    }
    const auto* data = model_.dialPayloads.data() + dial_.payloadOffset;
    for( int axis = 0; axis < 2; ++axis ){
      if( not std::isfinite(data[axis]) or data[axis] < 2. or data[axis] != std::floor(data[axis])
          or data[axis] > double(dial_.payloadSize) ){
        throw std::runtime_error("Surface axes require at least two knots and integer dimensions.");
      }
    }
    const auto nx = std::size_t(data[0]), ny = std::size_t(data[1]);
    // Guard the product as well as the subsequent coordinate reads.
    if( nx > (dial_.payloadSize - 2) / ny or 2 + nx + ny + nx * ny != dial_.payloadSize ){
      throw std::runtime_error("Surface dial dimensions do not match its payload.");
    }
    for( int axis = 0; axis < 2; ++axis ){
      const auto* values = data + 2 + (axis == 0 ? 0 : nx);
      const auto count = axis == 0 ? nx : ny;
      for( std::size_t i = 0; i < count; ++i ){
        if( not std::isfinite(values[i]) or (i != 0 and values[i] <= values[i - 1]) ){
          throw std::runtime_error("Surface coordinates must be finite and strictly increasing.");
        }
      }
    }
  }

  inline void validateTableDial(const PropagationView& model_, const BackendDialDescriptor& dial_) {
    if( not isTableDial(dial_.type) ){ return; }
    if( dial_.externalBlockIndex >= model_.externalWeightBlocks.size()
        or dial_.firstTableTerm > model_.tableTerms.size()
        or dial_.tableTermCount > model_.tableTerms.size() - dial_.firstTableTerm
        or (dial_.type == BackendDialType::Tabulated and dial_.tableTermCount != 2) ){
      throw std::runtime_error("Invalid backend table dial descriptor.");
    }
    const auto count = model_.externalWeightBlocks[dial_.externalBlockIndex].count;
    for( std::size_t i = 0; i < dial_.tableTermCount; ++i ){
      const auto& term = model_.tableTerms[dial_.firstTableTerm + i];
      if( term.index >= count or not std::isfinite(term.coefficient) ){
        throw std::runtime_error("Invalid backend table dial term.");
      }
    }
  }

  inline const double* getParameterValues(const PropagationInputs& inputs_) {
    return inputs_.parameters.values.data();
  }

  inline double evalDialResponse(const PropagationView& propagation_,
                                 const BackendDialDescriptor& dialRef_,
                                 const PropagationInputs& inputs_) {
    if( isTableDial(dialRef_.type) ){
      const auto* table = inputs_.externalWeights[dialRef_.externalBlockIndex].values;
      double response{0.};
      for( std::size_t i = 0; i < dialRef_.tableTermCount; ++i ){
        const auto& term = propagation_.tableTerms[dialRef_.firstTableTerm + i];
        response += term.coefficient * table[term.index];
      }
      return clampDialResponse(dialRef_, response);
    }
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
    const auto& weightRef = propagation_.eventWeights[eventRef_.resultIndex];
    double weight = 1.;
    for( std::size_t iDial = 0; iDial < weightRef.dialCount; ++iDial ){
      const auto& dial = propagation_.dials[propagation_.eventDialIndices[weightRef.firstDial + iDial]];
      weight *= evalDialResponse(propagation_, dial, inputs_);
    }
    if( propagation_.hasReweightCap and weight > propagation_.maxReweight ){ weight = propagation_.maxReweight; }
    return weightRef.baseWeight * weight;
  }

}

#endif // GUNDAM_BACKEND_DIAL_SEMANTICS_H
