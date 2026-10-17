#ifndef GUNDAM_EVENT_STATE_CACHE_H
#define GUNDAM_EVENT_STATE_CACHE_H

#include "PropagationInputs.h"

namespace Backends {
  // Execution-independent invalidation and publication of observable state.
  // Numerical buffers and bin-search strategies belong to the executor.
  class EventStateCache {
  public:
    void build(const PropagationView& model_);
    void prepare(const PropagationInputs& inputs_);
    void commit(const PropagationInputs& inputs_, const std::vector<std::uint8_t>& changedSamples_);
    void invalidate();
    [[nodiscard]] bool variableInputsChanged() const { return _variableInputsChanged_; }
    [[nodiscard]] bool needsUpdate(const PropagationInputs& inputs_) const;
    [[nodiscard]] bool hostVariableChanged(std::size_t index_, const PropagationInputs& inputs_) const;
    void publishRevisions(EventState& state_) const;

  private:
    const PropagationView* _model_{nullptr};
    std::vector<std::size_t> _variableDials_{};
    std::vector<double> _parameters_{};
    std::vector<double> _hostVariables_{};
    std::vector<std::uint64_t> _externalGenerations_{};
    std::vector<std::uint64_t> _dialRevisions_{};
    std::vector<std::uint64_t> _sampleRevisions_{};
    std::uint64_t _revision_{1};
    std::uint64_t _variableRevision_{1};
    bool _valid_{false};
    bool _variableInputsChanged_{true};
  };
}
#endif
