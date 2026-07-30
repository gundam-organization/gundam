#ifndef GUNDAM_BACKEND_DIAL_SEMANTICS_H
#define GUNDAM_BACKEND_DIAL_SEMANTICS_H

#include "BackendEngineView.h"
#include "ParameterSnapshot.h"

#include "CalculateCompactSpline.h"
#include "CalculateGeneralSpline.h"
#include "CalculateGraph.h"
#include "CalculateMonotonicSpline.h"
#include "CalculateUniformSpline.h"
#include "Logger.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace Backends::Semantics {

  inline double transformDialInput(const BackendDialInputView& inputRef_, double rawValue_) {
    if( not inputRef_.useMirror ){ return rawValue_; }

    double transformed = std::abs(std::fmod(
        rawValue_ - inputRef_.mirrorMin,
        2 * inputRef_.mirrorRange
    ));

    if( transformed > inputRef_.mirrorRange ){
      transformed -= 2 * inputRef_.mirrorRange;
      transformed = -transformed;
    }

    return transformed + inputRef_.mirrorMin;
  }

  inline double clampDialResponse(const BackendDialView& dialRef_, double response_) {
    if( dialRef_.hasMinResponse and response_ < dialRef_.minResponse ){ response_ = dialRef_.minResponse; }
    if( dialRef_.hasMaxResponse and response_ > dialRef_.maxResponse ){ response_ = dialRef_.maxResponse; }
    return response_;
  }

  inline double getParameterValue(const BackendDialInputView& inputRef_, const ParameterSnapshot& parameters_) {
    LogThrowIf(parameters_.empty(), "BackendDialSemantics requires a populated ParameterSnapshot.");
    return parameters_.values.at(inputRef_.parameterIndex);
  }

  inline const double* getDialPayload(const BackendPropagationView& propagation_, const BackendDialView& dialRef_) {
    return propagation_.dialPayloads.data() + dialRef_.payloadOffset;
  }

  template<typename RawInputProvider>
  inline double evalDialResponse(const BackendPropagationView& propagation_,
                                 const BackendDialView& dialRef_,
                                 RawInputProvider&& rawInputProvider_) {
    auto getInput = [&propagation_, &dialRef_, &rawInputProvider_](std::size_t iInput_){
      const auto& inputRef = propagation_.dialInputs.at(dialRef_.firstInput + iInput_);
      return transformDialInput(inputRef, rawInputProvider_(inputRef));
    };

    const double* payload = getDialPayload(propagation_, dialRef_);

    switch( dialRef_.type ){
      case BackendDialType::Norm:
        LogThrowIf(dialRef_.inputCount != 1, "Backend Norm dial expects exactly one input.");
        return clampDialResponse(dialRef_, getInput(0));

      case BackendDialType::Shift:
        LogThrowIf(dialRef_.payloadSize < 1, "Backend Shift dial payload is empty.");
        return clampDialResponse(dialRef_, payload[0]);

      case BackendDialType::CompactSpline: {
        LogThrowIf(dialRef_.payloadSize < 3, "Backend CompactSpline payload is too small.");
        double x = getInput(0);
        if( not dialRef_.allowExtrapolation ){
          x = std::clamp(x, payload[0], payload[0] + payload[1] * double(dialRef_.payloadSize - 3));
        }
        return clampDialResponse(dialRef_, CalculateCompactSpline(x, -1E20, 1E20, payload, int(dialRef_.payloadSize - 2)));
      }

      case BackendDialType::UniformSpline: {
        LogThrowIf(dialRef_.payloadSize < 4, "Backend UniformSpline payload is too small.");
        double x = getInput(0);
        if( not dialRef_.allowExtrapolation ){
          x = std::clamp(x, payload[0], payload[0] + payload[1] * double((dialRef_.payloadSize - 2) / 2 - 1));
        }
        return clampDialResponse(dialRef_, CalculateUniformSpline(x, -1E20, 1E20, payload, int(dialRef_.payloadSize)));
      }

      case BackendDialType::MonotonicSpline: {
        LogThrowIf(dialRef_.payloadSize < 3, "Backend MonotonicSpline payload is too small.");
        double x = getInput(0);
        if( not dialRef_.allowExtrapolation ){
          x = std::clamp(x, payload[0], payload[0] + payload[1] * double(dialRef_.payloadSize - 3));
        }
        return clampDialResponse(dialRef_, CalculateMonotonicSpline(x, -1E20, 1E20, payload, int(dialRef_.payloadSize - 2)));
      }

      case BackendDialType::GeneralSpline: {
        LogThrowIf(dialRef_.payloadSize < 5, "Backend GeneralSpline payload is too small.");
        double x = getInput(0);
        if( not dialRef_.allowExtrapolation ){
          x = std::clamp(x, payload[0], payload[0] + payload[1] * double((dialRef_.payloadSize - 2) / 3 - 1));
        }
        return clampDialResponse(dialRef_, CalculateGeneralSpline(x, -1E20, 1E20, payload, int(dialRef_.payloadSize)));
      }

      case BackendDialType::Graph: {
        LogThrowIf(dialRef_.payloadSize < 2, "Backend Graph payload is too small.");
        double x = getInput(0);
        if( not dialRef_.allowExtrapolation ){
          x = std::clamp(x, payload[1], payload[dialRef_.payloadSize - 1]);
        }
        return clampDialResponse(dialRef_, CalculateGraph(x, -1E20, 1E20, payload, int(dialRef_.payloadSize)));
      }
    }

    LogThrow("Unhandled backend dial type in BackendDialSemantics.");
  }

  inline double evalDialResponse(const BackendPropagationView& propagation_,
                                 const BackendDialView& dialRef_,
                                 const ParameterSnapshot& parameters_) {
    return evalDialResponse(
        propagation_,
        dialRef_,
        [&parameters_](const BackendDialInputView& inputRef_){ return getParameterValue(inputRef_, parameters_); }
    );
  }

  template<typename RawInputProvider>
  inline double evalEventWeight(const BackendPropagationView& propagation_,
                                const BackendEventView& eventRef_,
                                RawInputProvider&& rawInputProvider_) {
    double weight = eventRef_.baseWeight;

    for( std::size_t iDial = 0 ; iDial < eventRef_.dialCount ; iDial++ ){
      const auto& dialRef = propagation_.eventDials.at(eventRef_.firstDial + iDial);
      weight *= evalDialResponse(propagation_, dialRef, std::forward<RawInputProvider>(rawInputProvider_));
    }

    return weight;
  }

  inline double evalEventWeight(const BackendPropagationView& propagation_,
                                const BackendEventView& eventRef_,
                                const ParameterSnapshot& parameters_) {
    return evalEventWeight(
        propagation_,
        eventRef_,
        [&parameters_](const BackendDialInputView& inputRef_){ return getParameterValue(inputRef_, parameters_); }
    );
  }

}

#endif // GUNDAM_BACKEND_DIAL_SEMANTICS_H
