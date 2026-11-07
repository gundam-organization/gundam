#ifndef GUNDAM_DIAL_DEPENDENCIES_H
#define GUNDAM_DIAL_DEPENDENCIES_H

#include "EngineView.h"

namespace Backends {
  // Host-side dependency discovery only. Consumers retain their validation,
  // constant-dial handling, ordering, deduplication and cache policies.
  template<typename Parameter, typename External>
  void visitDialDependencies(const PropagationView& model_, const BackendDialDescriptor& dial_,
                             Parameter parameter_, External external_) {
    // Table producer inputs do not affect evaluation: publication of the
    // external block is the dependency, even when the producer has inputs.
    if( dial_.type == BackendDialType::ExternalWeight or isTableDial(dial_.type) ){
      external_(dial_.externalBlockIndex);
      return;
    }
    for( std::size_t i = dial_.firstInput; i < dial_.firstInput + dial_.inputCount; ++i ){
      parameter_(model_.dialInputs.at(i));
    }
  }
}
#endif
