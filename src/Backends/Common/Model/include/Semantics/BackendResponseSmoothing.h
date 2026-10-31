#ifndef GUNDAM_BACKEND_RESPONSE_SMOOTHING_H
#define GUNDAM_BACKEND_RESPONSE_SMOOTHING_H

#ifndef __METAL_VERSION__
#include "Semantics/BackendSemanticsQualifiers.h"
#define GUNDAM_RESPONSE_INLINE GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
#else
#define GUNDAM_RESPONSE_INLINE inline
#endif

namespace Backends {
  template<typename Scalar> struct ResponseSmoothing {
    Scalar minResponse{0}, maxResponse{0};
    Scalar minWidth{0}, maxWidth{0};
    Scalar minInverseWidth{0}, maxInverseWidth{0};
  };
}

namespace Backends::Semantics {
  // Called only outside the precomputed transition thresholds. With smoothing
  // disabled, the ordinary in-range path still consists of two comparisons.
  template<typename Scalar>
  GUNDAM_RESPONSE_INLINE Scalar processLowerResponse(Scalar value_, Scalar threshold_, ResponseSmoothing<Scalar> smoothing_) {
    if( smoothing_.minWidth == Scalar(0) ){ return threshold_; }
    if( value_ <= smoothing_.minResponse - smoothing_.minWidth ){ return smoothing_.minResponse; }
    const Scalar t = Scalar(0.5f) + (value_ - smoothing_.minResponse) * smoothing_.minInverseWidth;
    return smoothing_.minResponse + smoothing_.minWidth * (t * t * t * (Scalar(2) - t));
  }

  template<typename Scalar>
  GUNDAM_RESPONSE_INLINE Scalar processUpperResponse(Scalar value_, Scalar threshold_, ResponseSmoothing<Scalar> smoothing_) {
    if( smoothing_.maxWidth == Scalar(0) ){ return threshold_; }
    if( value_ >= smoothing_.maxResponse + smoothing_.maxWidth ){ return smoothing_.maxResponse; }
    const Scalar t = Scalar(0.5f) + (smoothing_.maxResponse - value_) * smoothing_.maxInverseWidth;
    return smoothing_.maxResponse - smoothing_.maxWidth * (t * t * t * (Scalar(2) - t));
  }

  template<typename Scalar, typename Descriptor>
  GUNDAM_RESPONSE_INLINE Scalar processResponse(Scalar value_, Descriptor descriptor_) {
    if( value_ < descriptor_.minResponse ){
      return processLowerResponse(value_, descriptor_.minResponse, descriptor_.smoothing);
    }
    else if( value_ > descriptor_.maxResponse ){
      return processUpperResponse(value_, descriptor_.maxResponse, descriptor_.smoothing);
    }
    return value_;
  }
}

#undef GUNDAM_RESPONSE_INLINE
#endif
