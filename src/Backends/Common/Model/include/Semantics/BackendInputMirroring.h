#ifndef GUNDAM_BACKEND_INPUT_MIRRORING_H
#define GUNDAM_BACKEND_INPUT_MIRRORING_H
#ifndef __METAL_VERSION__
#include "Semantics/BackendSemanticsQualifiers.h"
#define GUNDAM_MIRROR_INLINE GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
#else
#define GUNDAM_MIRROR_INLINE inline
#endif
namespace Backends::Semantics {
  // Input is the signed remainder in [-2*range, 2*range]. The caller retains
  // its native remainder operation and the existing piecewise knot convention.
  template<typename Scalar>
  GUNDAM_MIRROR_INLINE Scalar mirrorSlopeFromFolded(Scalar folded_, Scalar range_) {
    Scalar slope = folded_ < Scalar(0) ? Scalar(-1) : Scalar(1);
    const Scalar magnitude = folded_ < Scalar(0) ? -folded_ : folded_;
    return magnitude > range_ ? -slope : slope;
  }
}
#undef GUNDAM_MIRROR_INLINE
#endif
