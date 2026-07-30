#include "BackendEngineLayout.h"

#include "LikelihoodInterface.h"

void Backends::BackendEngineLayout::clear() {
  view.clear();
  bindings.clear();
}

void Backends::BackendEngineLayout::build(LikelihoodInterface& likelihoodInterface_) {
  clear();
  view.build(likelihoodInterface_);
  bindings.build(likelihoodInterface_);
}
