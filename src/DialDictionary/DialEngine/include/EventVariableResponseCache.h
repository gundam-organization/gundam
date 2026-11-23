/// @file EventVariableResponseCache.h
/// @brief Declares EventVariableResponseCache, which applies the dials that
///        change event variables and re-bins the events when needed.
/// @ingroup DialDictionary

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
///
/// A dial collection with `applyToVariable` changes an event variable
/// instead of the event weight (see DialCollection::getApplyToVariable() and
/// EventDialCache::applyVariableResponses()). When that variable is used by
/// a sample binning, events can move between bins, so the samples must use
/// dynamic binning. This class makes sure they do, and after each parameter
/// change it updates the variables and the event binning, but only when the
/// input of a variable dial changed.
///
/// @par Ownership
/// Each Propagator owns one by value. It keeps raw pointers to the input
/// buffers of the variable-dial collections.
///
/// @par Life cycle
/// 1. Propagator::configureImpl() and Propagator::initializeImpl(): prepare().
/// 2. Propagator::clearContent(): invalidate(), then prepare() again.
/// 3. Propagator::buildDialCache(): invalidate(), then build() once the
///    event-dial cache is built.
/// 4. Every propagation, right after DialManager::updateDialState()
///    (Propagator::applyParameters(), Propagator::reweightEvents()):
///    update().
/// @ingroup DialDictionary
class EventVariableResponseCache {
public:
  /// Discard input references before the dial collections are rebuilt.
  /// Preserve which samples were made dynamic during configuration.
  ///
  /// The next update() then applies the variable responses whatever the
  /// inputs.
  void invalidate();

  /// Register required variables and enable migration before event loading.
  /// Repeated calls preserve explicitly configured dynamic binning.
  ///
  /// For each enabled collection with `applyToVariable`, adds the variable
  /// to the event variables to load and turns on dynamic binning for every
  /// sample whose binning uses it; those samples are remembered as made
  /// dynamic for variable dials.
  /// @param sampleSet_ The samples; their dynamic-binning flag and the list
  ///        of event variables are modified.
  /// @param collections_ The dial collections of the same Propagator.
  void prepare(SampleSet& sampleSet_, const std::vector<DialCollection>& collections_);

  /// Bind to loaded dial inputs and force the next update.
  ///
  /// Records the current revision of each input buffer of each enabled
  /// variable-dial collection.
  /// @param collections_ The dial collections; their input buffers must not
  ///        move afterwards.
  void build(std::vector<DialCollection>& collections_);

  /// Apply variable responses, then rebuild bin contributions when needed.
  /// Explicitly dynamic samples are always rebinned when variable dials exist.
  ///
  /// Does nothing if no event has a variable dial
  /// (EventDialCache::hasVariableResponses()). Otherwise, if any recorded
  /// input buffer changed revision since the last update (or after
  /// invalidate()), calls EventDialCache::applyVariableResponses() for every
  /// entry, split between threads, then updates the binning of all samples.
  /// If nothing changed, only the samples whose dynamic binning was set in
  /// the configuration (not by prepare()) are rebinned.
  /// @param eventDialCache_ The Propagator's event-dial cache.
  /// @param sampleSet_ The Propagator's samples.
  /// @param threadPool_ Threads used for both steps.
  /// @pre The dial inputs were updated for the current parameters
  ///      (DialManager::updateDialState()).
  void update(EventDialCache& eventDialCache_, SampleSet& sampleSet_,
              GenericToolbox::ParallelWorker& threadPool_);

private:
  /// True after an update() that applied the responses, until invalidate().
  bool _variableStateValid_{false};
  /// Each input buffer of a variable-dial collection, with the revision it
  /// had at the last applied update.
  std::vector<std::pair<const DialInputBuffer*, std::uint64_t>> _variableInputRevisions_;
  /// Samples made dynamic solely for variable dials can retain their bin lists.
  ///
  /// One flag per sample: true if prepare() turned on its dynamic binning.
  std::vector<bool> _variableDialBinningSamples_;
};

#endif // GUNDAM_EVENT_VARIABLE_RESPONSE_CACHE_H
