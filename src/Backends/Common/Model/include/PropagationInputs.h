#ifndef GUNDAM_PROPAGATION_INPUTS_H
#define GUNDAM_PROPAGATION_INPUTS_H

#include "EngineView.h"
#include "ParameterSnapshot.h"

#include <stdexcept>

namespace Backends {

  struct ExternalWeightBlockView {
    const double* values{nullptr};
    std::size_t count{0};
    std::size_t destinationOffset{0};
    std::uint64_t generation{0};
  };

  // Borrowed host blocks: a backend must consume or copy their contents before
  // requestPropagation returns. Producers must not write during that call.
  struct PropagationInputs {
    ParameterSnapshot parameters{};
    // Optional per-dial Jacobians for external/table producers, with respect
    // to their transformed inputs. Copied by CPU before returning the token.
    std::vector<std::vector<double>> externalDialGradients{};
    // Optional sparse row indices, strictly increasing. When omitted, the
    // legacy representation indexes externalDialGradients by backend dial.
    std::vector<std::size_t> externalDialGradientIndices{};
    std::vector<ExternalWeightBlockView> externalWeights{};
    // Optional host edits to explicitly dynamic observables, in model order.
    // Empty keeps the state resident in the backend.
    std::vector<double> hostVariables{};
    bool forceBinningUpdate{false};
    // Optional revisions of unique input buffers used by observable dials.
    std::vector<std::uint64_t> variableInputRevisions{};

    void validate(const PropagationView& model_) const {
      if( not externalDialGradientIndices.empty() ){
        if( externalDialGradientIndices.size() != externalDialGradients.size() ){
          throw std::runtime_error("Backend sparse gradient row count mismatch.");
        }
        for( std::size_t i = 0; i < externalDialGradientIndices.size(); ++i ){
          const auto index = externalDialGradientIndices[i];
          if( index >= model_.dials.size() or (i > 0 and index <= externalDialGradientIndices[i-1]) ){
            throw std::runtime_error("Backend sparse gradient indices must be valid and strictly increasing.");
          }
          const auto type = model_.dials[index].type;
          if( type != BackendDialType::ExternalWeight and not isTableDial(type) ){
            throw std::runtime_error("Backend sparse gradients require external/table dials.");
          }
        }
      }
      if( not variableInputRevisions.empty() and variableInputRevisions.size() != model_.variableInputBufferCount ){
        throw std::runtime_error("Backend variable input revision count mismatch.");
      }
      if( not hostVariables.empty() and hostVariables.size() != model_.variableValues.size() ){
        throw std::runtime_error("Backend host variable buffer size mismatch.");
      }
      if( parameters.values.size() != model_.parameterCount ){
        throw std::runtime_error("Backend parameter snapshot size mismatch.");
      }
      if( externalWeights.size() != model_.externalWeightBlocks.size() ){
        throw std::runtime_error("Backend external weight block count mismatch.");
      }
      for( std::size_t iBlock = 0; iBlock < externalWeights.size(); ++iBlock ){
        const auto& block = externalWeights[iBlock];
        const auto& layout = model_.externalWeightBlocks[iBlock];
        if( block.count != layout.count or block.destinationOffset != layout.offset
            or (block.count != 0 and block.values == nullptr) or block.generation == 0 ){
          throw std::runtime_error("Backend external weight block is invalid or unpublished.");
        }
      }
    }
  };
}

#endif // GUNDAM_PROPAGATION_INPUTS_H
