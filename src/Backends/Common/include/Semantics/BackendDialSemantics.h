#ifndef GUNDAM_BACKEND_DIAL_SEMANTICS_H
#define GUNDAM_BACKEND_DIAL_SEMANTICS_H

#include "EngineView.h"
#include "ParameterSnapshot.h"
#include "Semantics/BackendDialSemanticsCore.h"

namespace Backends::Semantics {
  inline const double* getParameterValues(const ParameterSnapshot& parameters_) {
    GUNDAM_BACKEND_SEMANTICS_ASSERT(not parameters_.empty());
    return parameters_.values.data();
  }

  inline double evalDialResponse(const PropagationView& propagation_,
                                 const BackendDialDescriptor& dialRef_,
                                 const ParameterSnapshot& parameters_) {
    return evalDialResponse(
        dialRef_,
        propagation_.dialInputs.data(),
        propagation_.dialPayloads.data(),
        getParameterValues(parameters_)
    );
  }

  inline double evalEventWeight(const PropagationView& propagation_,
                                const EventView& eventRef_,
                                const ParameterSnapshot& parameters_) {
    return evalEventWeight(
        eventRef_.weight,
        propagation_.eventDials.data(),
        propagation_.dialInputs.data(),
        propagation_.dialPayloads.data(),
        getParameterValues(parameters_)
    );
  }

}

#endif // GUNDAM_BACKEND_DIAL_SEMANTICS_H
