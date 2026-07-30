#ifndef GUNDAM_BACKEND_DIAL_SEMANTICS_H
#define GUNDAM_BACKEND_DIAL_SEMANTICS_H

#include "BackendEngineView.h"
#include "ParameterSnapshot.h"
#include "Semantics/BackendDialSemanticsCore.h"

namespace Backends::Semantics {

  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  BackendDialInputCore toCore(const BackendDialInputView& inputRef_) {
    BackendDialInputCore out;
    out.parameterIndex = inputRef_.parameterIndex;
    out.useMirror = inputRef_.useMirror;
    out.mirrorMin = inputRef_.mirrorMin;
    out.mirrorRange = inputRef_.mirrorRange;
    return out;
  }

  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  BackendDialCore toCore(const BackendDialView& dialRef_) {
    BackendDialCore out;
    out.type = static_cast<std::uint8_t>(dialRef_.type);
    out.firstInput = dialRef_.firstInput;
    out.inputCount = dialRef_.inputCount;
    out.payloadOffset = dialRef_.payloadOffset;
    out.payloadSize = dialRef_.payloadSize;
    out.allowExtrapolation = dialRef_.allowExtrapolation;
    out.minResponse = dialRef_.minResponse;
    out.maxResponse = dialRef_.maxResponse;
    out.hasMinResponse = dialRef_.hasMinResponse;
    out.hasMaxResponse = dialRef_.hasMaxResponse;
    return out;
  }

  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  BackendEventCore toCore(const BackendEventView& eventRef_) {
    BackendEventCore out;
    out.baseWeight = eventRef_.baseWeight;
    out.firstDial = eventRef_.firstDial;
    out.dialCount = eventRef_.dialCount;
    return out;
  }

  inline const double* getParameterValues(const ParameterSnapshot& parameters_) {
    GUNDAM_BACKEND_SEMANTICS_ASSERT(not parameters_.empty());
    return parameters_.values.data();
  }

  inline double evalDialResponse(const BackendPropagationView& propagation_,
                                 const BackendDialView& dialRef_,
                                 const ParameterSnapshot& parameters_) {
    const BackendDialCore dialCore = toCore(dialRef_);
    const double* payload = getDialPayload(propagation_.dialPayloads.data(), dialCore);

    if( dialCore.type == 1 ){
      return evalDialResponseFromInput(dialCore, 0., payload);
    }

    GUNDAM_BACKEND_SEMANTICS_ASSERT(dialCore.inputCount >= 1);
    const auto& inputRef = propagation_.dialInputs.at(dialRef_.firstInput);
    const double inputValue = transformDialInput(toCore(inputRef), loadParameterValue(toCore(inputRef), getParameterValues(parameters_)));
    return evalDialResponseFromInput(dialCore, inputValue, payload);
  }

  inline double evalEventWeight(const BackendPropagationView& propagation_,
                                const BackendEventView& eventRef_,
                                const ParameterSnapshot& parameters_) {
    const BackendEventCore eventCore = toCore(eventRef_);
    double weight = eventCore.baseWeight;
    for( std::size_t iDial = 0 ; iDial < eventCore.dialCount ; iDial++ ){
      weight *= evalDialResponse(propagation_, propagation_.eventDials.at(eventRef_.firstDial + iDial), parameters_);
    }
    return weight;
  }

}

#endif // GUNDAM_BACKEND_DIAL_SEMANTICS_H
