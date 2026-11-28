#ifndef GUNDAM_EVENT_VARIABLE_RESPONSE_CACHE_H
#define GUNDAM_EVENT_VARIABLE_RESPONSE_CACHE_H

#include <cstdint>
#include <utility>
#include <vector>

class DialCollection;
class DialInputBuffer;
class EventDialCache;
class SampleSet;
namespace GenericToolbox { class ParallelWorker; }

/// Prepare variable dials and retain their event variables and bin contributions
/// until a dependent input changes. Does not own the referenced input buffers.
class EventVariableResponseCache {
public:
  /// Discard input references before the dial collections are rebuilt.
  /// Preserve which samples were made dynamic during configuration.
  void invalidate();
  void invalidateState(){ _variableStateValid_ = false; }
  [[nodiscard]] bool isAutomaticBinningSample(std::size_t sample_) const {
    return sample_ < _variableDialBinningSamples_.size() and _variableDialBinningSamples_[sample_];
  }

  /// Register required variables and enable migration before event loading.
  /// Repeated calls preserve explicitly configured dynamic binning.
  void prepare(SampleSet& sampleSet_, const std::vector<DialCollection>& collections_);

  /// Bind to loaded dial inputs and force the next update.
  void build(std::vector<DialCollection>& collections_);

  /// Apply variable responses, then rebuild bin contributions when needed.
  /// Explicitly dynamic samples are always rebinned when variable dials exist.
  void update(EventDialCache& eventDialCache_, SampleSet& sampleSet_,
              GenericToolbox::ParallelWorker& threadPool_);

private:
  bool _variableStateValid_{false};
  std::vector<std::pair<const DialInputBuffer*, std::uint64_t>> _variableInputRevisions_;
  // Samples made dynamic solely for variable dials can retain their bin lists.
  std::vector<bool> _variableDialBinningSamples_;
};

#endif // GUNDAM_EVENT_VARIABLE_RESPONSE_CACHE_H
