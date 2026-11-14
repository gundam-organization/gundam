#ifndef GUNDAM_BACKEND_VARIABLE_RESPONSE_H
#define GUNDAM_BACKEND_VARIABLE_RESPONSE_H

namespace Backends::Semantics {
  // Composition for CPU, CUDA and Metal descriptors, using packed dial indices.
  template<typename Target, typename Indices, typename Response>
#ifdef __CUDACC__
  __host__ __device__
#endif
  inline auto evalVariableResponse(Target target_, Indices indices_, Response response_) -> decltype(target_.referenceValue) {
    auto value = target_.overrideDial == decltype(target_.overrideDial)(-1) ? target_.referenceValue : response_(target_.overrideDial);
    for( decltype(target_.firstMultiplyDial) i = target_.firstMultiplyDial; i < target_.firstMultiplyDial + target_.multiplyDialCount; ++i ){
      value *= response_(indices_[i]);
    }
    for( decltype(target_.firstAddDial) i = target_.firstAddDial; i < target_.firstAddDial + target_.addDialCount; ++i ){
      value += response_(indices_[i]);
    }
    return value;
  }
}

#endif
