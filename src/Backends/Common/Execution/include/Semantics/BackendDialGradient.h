#ifndef GUNDAM_BACKEND_DIAL_GRADIENT_H
#define GUNDAM_BACKEND_DIAL_GRADIENT_H

#include "Semantics/BackendDialSemanticsCore.h"

namespace Backends::Semantics {
  // Single parameter weight dials only. Callers validate the dial type and
  // reject response smoothing before invoking this portable derivative.
  GUNDAM_BACKEND_HOST_DEVICE inline double evalWeightDialGradient(
      const BackendDialDescriptor& dial_, const BackendDialInputDescriptor& input_,
      const double* payload_, double rawInput_) {
    double inputGradient = 1.;
    if( input_.useMirror ){
      if( input_.mirrorRange <= 0. ){ return 0.; }
      const double folded = ::fmod(rawInput_ - input_.mirrorMin, 2. * input_.mirrorRange);
      inputGradient = folded < 0. ? -1. : 1.;
      if( ::fabs(folded) > input_.mirrorRange ){ inputGradient = -inputGradient; }
    }
    const double x = transformDialInput(input_, rawInput_);
    double low = 0., high = 0., derivative = 0.;
    switch( dial_.type ){
      case BackendDialType::Norm: derivative = 1.; break;
      case BackendDialType::Shift: return 0.;
      case BackendDialType::Graph:
        low = payload_[1]; high = payload_[dial_.payloadSize - 1];
        derivative = CalculateGraphGradient(x, -1E20, 1E20, payload_, int(dial_.payloadSize));
        break;
      case BackendDialType::CompactSpline:
        low = payload_[0]; high = low + payload_[1] * double(dial_.payloadSize - 3);
        derivative = CalculateCompactSplineGradient(x, -1E20, 1E20, payload_, int(dial_.payloadSize - 2));
        break;
      case BackendDialType::UniformSpline:
        low = payload_[0]; high = low + payload_[1] * double((dial_.payloadSize - 2) / 2 - 1);
        derivative = CalculateUniformSplineGradient(x, -1E20, 1E20, payload_, int(dial_.payloadSize));
        break;
      case BackendDialType::MonotonicSpline:
        low = payload_[0]; high = low + payload_[1] * double(dial_.payloadSize - 3);
        derivative = CalculateMonotonicSplineGradient(x, -1E20, 1E20, payload_, int(dial_.payloadSize - 2));
        break;
      case BackendDialType::GeneralSpline:
        low = payload_[0]; high = low + payload_[1] * double((dial_.payloadSize - 2) / 3 - 1);
        derivative = CalculateGeneralSplineGradient(x, -1E20, 1E20, payload_, int(dial_.payloadSize));
        break;
      default: GUNDAM_BACKEND_SEMANTICS_ASSERT(false); return 0.;
    }
    if( dial_.type != BackendDialType::Norm and not dial_.allowExtrapolation
        and (x <= low or x >= high) ){ return 0.; }
    auto unclamped = dial_;
    unclamped.hasMinResponse = false;
    unclamped.hasMaxResponse = false;
    const double response = evalDialResponseFromInput(unclamped, x, payload_);
    if( (dial_.hasMinResponse and response < dial_.minResponse)
        or (dial_.hasMaxResponse and response > dial_.maxResponse) ){ return 0.; }
    return derivative * inputGradient;
  }
}
#endif
