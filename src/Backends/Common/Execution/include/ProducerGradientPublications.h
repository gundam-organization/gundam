#ifndef GUNDAM_PRODUCER_GRADIENT_PUBLICATIONS_H
#define GUNDAM_PRODUCER_GRADIENT_PUBLICATIONS_H
#include "PropagationInputs.h"

#include <algorithm>
#include <cmath>

namespace Backends {
  // Inputs must have passed PropagationInputs::validate; producer indices are
  // valid and strictly increasing, as prepared by each backend at build time.
  // Rows are borrowed for the callback only. A null row means no publication;
  // an empty but published row is valid for a zero-input producer.
  // Value-only propagation is permitted without valid Jacobians. Callers copy
  // valid/invalid publication state on every request and reject on gradient use.
  template<typename Consumer>
  void visitProducerGradientPublications(const PropagationView& model_, const PropagationInputs& inputs_,
      const std::vector<std::size_t>& producerDials_, Consumer consume_) {
    const bool sparse = not inputs_.externalDialGradientIndices.empty();
    const bool dense = inputs_.externalDialGradients.size() == model_.dials.size();
    std::size_t position = 0;
    for( std::size_t i = 0; i < producerDials_.size(); ++i ){
      const auto index = producerDials_[i];
      while( sparse and position < inputs_.externalDialGradientIndices.size()
             and inputs_.externalDialGradientIndices[position] < index ) ++position;
      const bool published = sparse ? (position < inputs_.externalDialGradientIndices.size()
          and inputs_.externalDialGradientIndices[position] == index) : dense;
      const auto* row = published ? &inputs_.externalDialGradients[sparse ? position : index] : nullptr;
      const bool valid = row and row->size() == model_.dials[index].inputCount
          and std::all_of(row->begin(), row->end(), [](double value_){ return std::isfinite(value_); });
      consume_(i, index, row, valid);
    }
  }
}
#endif
