#ifndef GUNDAM_GRADIENT_TYPES_H
#define GUNDAM_GRADIENT_TYPES_H
#ifndef __METAL_VERSION__
#include "DevicePropagationTypes.h"
#endif
namespace Backends {
  template<typename S> struct GradientDial {
    DeviceIndex type{}, parameter{}, payload{}, payloadSize{};
    DeviceIndex extrapolate{}, mirror{}, hasMin{}, hasMax{};
    S mirrorMin{}, mirrorRange{}, minResponse{}, maxResponse{};
  };
  template<typename S> struct GradientEvent {
    DeviceIndex first{}, count{}, bin{}, result{};
    S base{};
  };
  template<typename S> struct GradientWork {
    DeviceIndex events{}, dials{}, bins{}, parameters{};
    S cap{};
    DeviceIndex useCap{}, reserved{};
  };
#ifndef __METAL_VERSION__
  static_assert(sizeof(GradientDial<double>) == 64);
  static_assert(sizeof(GradientEvent<double>) == 24);
  static_assert(sizeof(GradientWork<double>) == 32);
  static_assert(sizeof(GradientDial<ObservableScalar>) == sizeof(GradientDial<double>));
  static_assert(sizeof(GradientEvent<ObservableScalar>) == sizeof(GradientEvent<double>));
  static_assert(sizeof(GradientWork<ObservableScalar>) == sizeof(GradientWork<double>));
#endif
}
#endif
