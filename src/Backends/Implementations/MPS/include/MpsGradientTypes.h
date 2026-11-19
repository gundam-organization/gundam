#ifndef GUNDAM_MPS_GRADIENT_TYPES_H
#define GUNDAM_MPS_GRADIENT_TYPES_H
#ifndef __METAL_VERSION__
#include "GradientTypes.h"
#endif
namespace Backends {
  // One directional derivative per root input; repeated parameter bindings are
  // reduced together. Each direction owns its composite/formula scratch frame.
  struct MpsGradientInput { DeviceIndex dial, input, scratch, external; };
  struct MpsGradientEvent {
    DeviceIndex first, count, result, firstContribution;
    ObservableScalar base;
  };
#ifndef __METAL_VERSION__
  static_assert(sizeof(MpsGradientInput) == 16);
  static_assert(sizeof(MpsGradientEvent) == 24);
#endif
}
#endif
