#ifndef GUNDAM_MPS_WEIGHT_TYPES_H
#define GUNDAM_MPS_WEIGHT_TYPES_H

#ifndef __METAL_VERSION__
#include "DevicePropagationTypes.h"
#endif

namespace Backends {
  // The low two flag bits retain extrapolation/cache semantics. Remaining bits
  // address an optional smoothing record; zero means hard bounds only.
  struct MpsSplineDialDescriptor {
    DeviceIndex parameterIndex, splineOffset, splineSize, flags;
    float minResponse, maxResponse;
  };
  struct MpsNormDialOccurrence {
    DeviceIndex parameterIndex;
    float minResponse, maxResponse;
    DeviceIndex smoothingIndex;
  };
  struct MpsExternalDialOccurrence {
    DeviceIndex weightIndex;
    float minResponse, maxResponse;
    DeviceIndex smoothingIndex;
  };
#ifndef __METAL_VERSION__
  static_assert(sizeof(MpsSplineDialDescriptor) == 24);
  static_assert(sizeof(MpsNormDialOccurrence) == 16);
  static_assert(sizeof(MpsExternalDialOccurrence) == 16);
#endif
}
#endif
