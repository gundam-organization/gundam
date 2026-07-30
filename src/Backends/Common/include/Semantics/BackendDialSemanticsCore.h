#ifndef GUNDAM_BACKEND_DIAL_SEMANTICS_CORE_H
#define GUNDAM_BACKEND_DIAL_SEMANTICS_CORE_H

#include "Semantics/BackendSemanticsQualifiers.h"

#include "CalculateCompactSpline.h"
#include "CalculateGeneralSpline.h"
#include "CalculateGraph.h"
#include "CalculateMonotonicSpline.h"
#include "CalculateUniformSpline.h"

#include <cstddef>
#include <cstdint>
#include <cmath>

namespace Backends::Semantics {

  struct BackendDialInputCore {
    std::size_t parameterIndex{std::size_t(-1)};
    bool useMirror{false};
    double mirrorMin{0};
    double mirrorRange{0};
  };

  struct BackendDialCore {
    std::uint8_t type{0};
    std::size_t firstInput{0};
    std::size_t inputCount{0};
    std::size_t payloadOffset{0};
    std::size_t payloadSize{0};
    bool allowExtrapolation{false};
    double minResponse{0};
    double maxResponse{0};
    bool hasMinResponse{false};
    bool hasMaxResponse{false};
  };

  struct BackendEventCore {
    double baseWeight{1};
    std::size_t firstDial{0};
    std::size_t dialCount{0};
  };

  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double clampValue(double value_, double low_, double high_) {
    return value_ < low_ ? low_ : (value_ > high_ ? high_ : value_);
  }

  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double transformDialInput(const BackendDialInputCore& inputRef_, double rawValue_) {
    if( not inputRef_.useMirror ){ return rawValue_; }

    double transformed = ::fabs(::fmod(
        rawValue_ - inputRef_.mirrorMin,
        2 * inputRef_.mirrorRange
    ));

    if( transformed > inputRef_.mirrorRange ){
      transformed -= 2 * inputRef_.mirrorRange;
      transformed = -transformed;
    }

    return transformed + inputRef_.mirrorMin;
  }

  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double clampDialResponse(const BackendDialCore& dialRef_, double response_) {
    if( dialRef_.hasMinResponse and response_ < dialRef_.minResponse ){ response_ = dialRef_.minResponse; }
    if( dialRef_.hasMaxResponse and response_ > dialRef_.maxResponse ){ response_ = dialRef_.maxResponse; }
    return response_;
  }

  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double loadParameterValue(const BackendDialInputCore& inputRef_, const double* parameterValues_) {
    GUNDAM_BACKEND_SEMANTICS_ASSERT(parameterValues_ != nullptr);
    return parameterValues_[inputRef_.parameterIndex];
  }

  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  const double* getDialPayload(const double* dialPayloads_, const BackendDialCore& dialRef_) {
    GUNDAM_BACKEND_SEMANTICS_ASSERT(dialPayloads_ != nullptr or dialRef_.payloadSize == 0);
    return dialPayloads_ + dialRef_.payloadOffset;
  }

  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double evalDialResponseFromInput(const BackendDialCore& dialRef_,
                                   double inputValue_,
                                   const double* payload_) {
    switch( dialRef_.type ){
      case 0:
        GUNDAM_BACKEND_SEMANTICS_ASSERT(dialRef_.inputCount == 1);
        return clampDialResponse(dialRef_, inputValue_);

      case 1:
        GUNDAM_BACKEND_SEMANTICS_ASSERT(dialRef_.payloadSize >= 1);
        return clampDialResponse(dialRef_, payload_[0]);

      case 2: {
        GUNDAM_BACKEND_SEMANTICS_ASSERT(dialRef_.payloadSize >= 3);
        double x = inputValue_;
        if( not dialRef_.allowExtrapolation ){
          x = clampValue(x, payload_[0], payload_[0] + payload_[1] * double(dialRef_.payloadSize - 3));
        }
        return clampDialResponse(dialRef_, CalculateCompactSpline(x, -1E20, 1E20, payload_, int(dialRef_.payloadSize - 2)));
      }

      case 3: {
        GUNDAM_BACKEND_SEMANTICS_ASSERT(dialRef_.payloadSize >= 4);
        double x = inputValue_;
        if( not dialRef_.allowExtrapolation ){
          x = clampValue(x, payload_[0], payload_[0] + payload_[1] * double((dialRef_.payloadSize - 2) / 2 - 1));
        }
        return clampDialResponse(dialRef_, CalculateUniformSpline(x, -1E20, 1E20, payload_, int(dialRef_.payloadSize)));
      }

      case 4: {
        GUNDAM_BACKEND_SEMANTICS_ASSERT(dialRef_.payloadSize >= 3);
        double x = inputValue_;
        if( not dialRef_.allowExtrapolation ){
          x = clampValue(x, payload_[0], payload_[0] + payload_[1] * double(dialRef_.payloadSize - 3));
        }
        return clampDialResponse(dialRef_, CalculateMonotonicSpline(x, -1E20, 1E20, payload_, int(dialRef_.payloadSize - 2)));
      }

      case 5: {
        GUNDAM_BACKEND_SEMANTICS_ASSERT(dialRef_.payloadSize >= 5);
        double x = inputValue_;
        if( not dialRef_.allowExtrapolation ){
          x = clampValue(x, payload_[0], payload_[0] + payload_[1] * double((dialRef_.payloadSize - 2) / 3 - 1));
        }
        return clampDialResponse(dialRef_, CalculateGeneralSpline(x, -1E20, 1E20, payload_, int(dialRef_.payloadSize)));
      }

      case 6: {
        GUNDAM_BACKEND_SEMANTICS_ASSERT(dialRef_.payloadSize >= 2);
        double x = inputValue_;
        if( not dialRef_.allowExtrapolation ){
          x = clampValue(x, payload_[1], payload_[dialRef_.payloadSize - 1]);
        }
        return clampDialResponse(dialRef_, CalculateGraph(x, -1E20, 1E20, payload_, int(dialRef_.payloadSize)));
      }
    }

    GUNDAM_BACKEND_SEMANTICS_ASSERT(false);
    return 1.;
  }

  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double evalDialResponse(const BackendDialCore& dialRef_,
                          const BackendDialInputCore* dialInputs_,
                          const double* dialPayloads_,
                          const double* parameterValues_) {
    GUNDAM_BACKEND_SEMANTICS_ASSERT(dialInputs_ != nullptr or dialRef_.inputCount == 0);
    const double* payload = getDialPayload(dialPayloads_, dialRef_);

    if( dialRef_.type == 1 ){
      return evalDialResponseFromInput(dialRef_, 0., payload);
    }

    GUNDAM_BACKEND_SEMANTICS_ASSERT(dialRef_.inputCount >= 1);
    const auto& inputRef = dialInputs_[dialRef_.firstInput];
    const double inputValue = transformDialInput(inputRef, loadParameterValue(inputRef, parameterValues_));
    return evalDialResponseFromInput(dialRef_, inputValue, payload);
  }

  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double evalEventWeight(const BackendEventCore& eventRef_,
                         const BackendDialCore* eventDials_,
                         const BackendDialInputCore* dialInputs_,
                         const double* dialPayloads_,
                         const double* parameterValues_) {
    GUNDAM_BACKEND_SEMANTICS_ASSERT(eventDials_ != nullptr or eventRef_.dialCount == 0);

    double weight = eventRef_.baseWeight;
    for( std::size_t iDial = 0 ; iDial < eventRef_.dialCount ; iDial++ ){
      const auto& dialRef = eventDials_[eventRef_.firstDial + iDial];
      weight *= evalDialResponse(dialRef, dialInputs_, dialPayloads_, parameterValues_);
    }
    return weight;
  }

}

#endif // GUNDAM_BACKEND_DIAL_SEMANTICS_CORE_H
