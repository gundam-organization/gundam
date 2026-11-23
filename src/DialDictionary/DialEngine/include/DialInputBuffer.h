//
// Created by Adrien Blanchet on 29/11/2022.
//

/// @file DialInputBuffer.h
/// @brief Declares DialInputBuffer, the cache of parameter values that dials
///        read when they are evaluated.
/// @ingroup DialDictionary

#ifndef GUNDAM_DIAL_INPUT_BUFFER_H
#define GUNDAM_DIAL_INPUT_BUFFER_H

#include "ParameterSet.h"

#include <vector>
#include <utility>
#include <cstdint>


/// Caches the current values of the fit parameters that one or more dials
/// read, and flags when any of them changed.
///
/// A dial is a response function of one or more fit parameters (DialBase).
/// A dial does not read Parameter objects: DialBase::evalResponse() reads the
/// plain values stored here (getInputBuffer()), in the order the parameter
/// references were added. update() copies the current parameter values into
/// this cache, applies the optional mirroring, and raises the
/// "update requested" flag, and increases a revision counter (getRevision()),
/// only when at least one value changed. EventDialCache compares the
/// revision with the one it saw at the last evaluation, so that dials whose
/// inputs did not change are not evaluated again.
///
/// @par Ownership
/// Each DialCollection owns its buffers by value: either one buffer shared by
/// all its dials, or (for a set-wide collection with no `dialInputList`) one
/// buffer per dial, where dial `i` reads parameter `i` of the set. Each
/// DialInterface keeps a non-owning pointer to its buffer. A buffer does not
/// own the parameters: it keeps a pointer to the ParametersManager's list of
/// ParameterSet objects and indices into it. CompositeDial and EventFormula
/// also fill temporary buffers with computed values (setInputValues()); those
/// have no parameter references.
///
/// @par Life cycle
/// 1. DialCollection calls setParSetRef() and addParameterReference(), once
///    per input, in the order the dial reads them.
/// 2. initialise() fixes the number of inputs and checks the indices
///    (DialCollection::setupDialInterfaceReferences()).
/// 3. Every likelihood evaluation: update(), through
///    DialManager::updateDialState() and DialCollection::updateInputBuffers().
/// 4. invalidateBuffers() makes the next update() report a change; called
///    through DialManager::invalidateInputBuffers() at the end of
///    Propagator::buildDialCache().
///
/// @par Thread safety
/// update() and invalidateBuffers() write the cache; they run on one thread
/// while no dial is evaluated. Several threads may read the cache at the
/// same time while no update runs, as during Propagator::reweightEvents().
///
/// @see DialInterface, DialCollection, EventDialCache::DialResponseCache
/// @ingroup DialDictionary
class DialInputBuffer {

  /// DialInputBuffer is in charge of keeping the "input" parameter values
  /// for the evaluation of a collection of dials.
  ///
  /// The buffer is updated in single thread before the evaluation of
  /// individual dials is done. Collection buffers only reference Parameters,
  /// not Event variables. Composite evaluations use separate call-local
  /// buffers populated with intermediate values via setInputValues().
  /// @class DialInputBuffer

public:
  // definition of data structs

  /// Identifies one fit parameter that the dial reads, plus the optional
  /// mirroring applied to its value.
  ///
  /// The parameter is found by position: `parSetIndex` in the ParameterSet
  /// list given to setParSetRef(), then `parIndex` in that set's
  /// ParameterSet::getParameterList(). DialCollection fills one per input
  /// and passes it to addParameterReference().
  struct ParameterReference{
    /// Position of the ParameterSet in the list given to setParSetRef(); -1
    /// until set.
    int parSetIndex{-1};
    /// Position of the Parameter in ParameterSet::getParameterList(); -1 until
    /// set.
    int parIndex{-1};
    int bufferIndex{-1}; ///< Slot of this input in the value cache; set by addParameterReference().

    // it costs less memory to have only one vector with two potentially useless double (2*8bytes)
    // than an empty vector (24 bytes)
    /// Mirror window of one input parameter.
    ///
    /// With mirroring, update() folds the parameter value back into
    /// [`minValue`, `minValue + range`] by reflecting it at both edges, so the
    /// dial only ever sees values inside that window.
    /// DialCollection::setupDialInterfaceReferences() sets both members from
    /// the `mirrorLowEdge` and `mirrorHighEdge` options when `useMirrorDial`
    /// is true.
    struct MirrorEdges{
      /// Lower edge of the window, in parameter units; NaN means no mirroring.
      double minValue{std::nan("unset")};
      /// Width of the window (upper edge minus lower edge); NaN when unset.
      double range{std::nan("unset")};
    };
    /// Mirroring applied to this input; unset (NaN) by default, meaning the
    /// parameter value is used as is.
    MirrorEdges mirrorEdges{};

    /// Looks up the ParameterSet of this input.
    /// @param parSetListPtr_ List of parameter sets the indices refer to.
    /// @return The set at `parSetIndex`. No bounds check.
    [[nodiscard]] auto& getParameterSet(std::vector<ParameterSet>* parSetListPtr_) const { return (*parSetListPtr_)[parSetIndex]; }
    /// Looks up the Parameter of this input.
    /// @param parSetListPtr_ List of parameter sets the indices refer to.
    /// @return The parameter at `parIndex` in the set at `parSetIndex`. No
    ///         bounds check.
    [[nodiscard]] auto& getParameter(std::vector<ParameterSet>* parSetListPtr_) const { return this->getParameterSet(parSetListPtr_).getParameterList()[parIndex]; }
  };

public:
  /// Creates a buffer with no inputs. Call setParSetRef(),
  /// addParameterReference() and initialise() before use.
  DialInputBuffer() = default;

  /// Tell the input buffer about the global vector of fit parameter sets.
  /// This is required, so it must be set before the DialInputBuffer can
  /// be used.
  /// @param parSetRef_ Non-owning pointer to the ParametersManager's list of
  ///        parameter sets. The list must not move while the buffer is used.
  void setParSetRef(std::vector<ParameterSet> *parSetRef_){ _parSetListPtr_ = parSetRef_; }

  // const getters

  /// Tells whether the last update() found a changed input value.
  /// @return True if at least one cached value changed during the last
  ///         update(); also true before the first update().
  /// @note ExternalWeightWorker::updateWeights() uses it to skip unchanged
  ///       inputs. EventDialCache checks getRevisionPtr() instead.
  [[nodiscard]] auto isDialUpdateRequested() const{ return _isDialUpdateRequested_; }

  /// Gives the revision counter of the cached values.
  ///
  /// The counter increases whenever the values may have changed: in update()
  /// when an input changed, in invalidateBuffers() and in setInputValues().
  /// EventVariableResponseCache compares it with the value it saw last.
  /// @return The current revision; 0 for a new buffer.
  [[nodiscard]] auto getRevision() const{ return _revision_; }

  /// Gives the address of the revision counter.
  ///
  /// EventDialCache::DialResponseCache stores this pointer so that it can
  /// check, for every event, whether the inputs changed since the cached
  /// response was computed, without going through the buffer.
  /// @return Pointer to `_revision_`. It stays valid as long as this buffer
  ///         is neither moved nor destroyed.
  [[nodiscard]] const std::uint64_t* getRevisionPtr() const{ return &_revision_; }

  /// Gives the address of the "update requested" flag.
  /// @return Pointer to the flag. It stays valid as long as this buffer is
  ///         neither moved nor destroyed.
  /// @note Not called in the current code: EventDialCache::DialResponseCache
  ///       now keeps getRevisionPtr() instead.
  [[nodiscard]] auto isDialUpdateRequestedPtr() const{ return &_isDialUpdateRequested_; }
  /// Gives the number of inputs, as fixed by initialise(). Cache::Manager
  /// uses it to loop over the parameters of a dial.
  /// @return Number of cached values; 0 before initialise().
  [[nodiscard]] auto getBufferSize() const{ return _inputArraySize_; }
  /// Gives the number of inputs.
  /// @return Before initialise(), the number of parameter references added;
  ///         afterwards, getBufferSize(), which also counts the values of a
  ///         buffer filled by setInputValues().
  [[nodiscard]] auto getInputSize() const{ return _isInitialized_ ? size_t(_inputArraySize_) : _inputParameterReferenceList_.size(); }
  /// Gives read-only access to the cached input values.
  /// @return One value per input, in the order of addParameterReference(),
  ///         after mirroring. This is what each DialBase::evalResponse()
  ///         implementation reads.
  [[nodiscard]] auto& getInputBuffer() const { return _inputBuffer_; }
  /// Gives read-only access to the parameter references.
  /// @return One ParameterReference per input, in input order.
  [[nodiscard]] auto& getInputParameterIndicesList() const{ return _inputParameterReferenceList_; }

  // mutable getters

  /// Function that allow to tweak the buffer from the inside. Used for
  /// individual spline evaluation.
  /// @return The cached input values. Writing here does not change the
  ///         "update requested" flag.
  /// @note Current callers: TabulatedDialFactory::updateTable() and
  ///       KrigedDialFactory::updateTable() (they only read the values, but
  ///       through a non-const buffer), and
  ///       EventTreeWriter::writeEventsTemplate(). The latter writes trial
  ///       values into a copy of the buffer and then calls
  ///       DialInterface::evalResponse(), which reads the original buffer,
  ///       so the trial values are not used there.
  auto& getInputBuffer(){ return _inputBuffer_; }
  /// Gives mutable access to the parameter references.
  /// DialCollection::setupDialInterfaceReferences() uses it to set the
  /// mirror edges of every input.
  /// @return One ParameterReference per input, in input order.
  /// @warning Do not add or remove references here after initialise():
  ///          update() writes cache slot `bufferIndex`, and the cache size is
  ///          fixed by initialise().
  auto& getInputParameterIndicesList(){ return _inputParameterReferenceList_; }

  // core

  /// Sets every cached input value to NaN, so that the next update() reports
  /// a change and every dial using this buffer is evaluated again.
  ///
  /// Called through DialCollection::invalidateCachedInputBuffers() by
  /// DialManager::invalidateInputBuffers(), at the end of
  /// Propagator::buildDialCache().
  /// It also increases the revision counter (getRevision()).
  /// @note Does not change the "update requested" flag itself. A dial
  ///       evaluated before the next update() reads NaN inputs.
  void invalidateBuffers();

  /// Make sure everything is ready for use
  ///
  /// Fixes the number of inputs to the number of references added, sizes
  /// the value cache (new slots are NaN), and checks that every reference
  /// points to an existing parameter set and parameter. Afterwards,
  /// addParameterReference() is refused.
  /// @pre setParSetRef() was called with a non-null pointer; otherwise the
  ///      job stops with an error. An out-of-range index also stops the job.
  /// @note DialCollection::setupDialInterfaceReferences() calls it, at
  ///       DialCollection::initialize() and again after each
  ///       DialCollection::resizeContainers(), so it may run several times.
  ///       Later calls keep the values already cached.
  void initialise();

  /// Update the buffer to flag if any parameter has changed, and apply any
  /// mirroring to the parameter values.
  ///
  /// For each input, reads Parameter::getParameterValue(), folds it into the
  /// mirror window if one is set (see ParameterReference::MirrorEdges), and
  /// stores it. The "update requested" flag ends true if at least one stored
  /// value changed (exact comparison), false otherwise; when it is true, the
  /// revision counter is increased as well.
  ///
  /// Called once per likelihood evaluation, on one thread, by
  /// DialCollection::updateInputBuffers() from DialManager::updateDialState().
  /// @pre initialise() has been called.
  void update();

  /// Call-local inputs for composed evaluations; no parameter references.
  ///
  /// Replaces the cached values with `values_` and marks the buffer as
  /// initialised and changed. CompositeDial and EventFormula use it on a
  /// temporary buffer to pass computed values to the dial they evaluate.
  /// @param values_ Input values, in the order the dial reads them.
  /// @pre The buffer has no parameter references; otherwise it throws.
  void setInputValues(std::vector<double> values_);

  // nested getters

  /// Gives the ParameterSet that one input belongs to.
  /// @param iInput_ Input position (0-based, order of addParameterReference()).
  /// @return The parameter set. No bounds check.
  /// @pre setParSetRef() has been called.
  [[nodiscard]] auto& getParameterSet(int iInput_) const{ return _inputParameterReferenceList_[iInput_].getParameterSet(_parSetListPtr_); }
  /// Gives the Parameter read by one input. Used by Cache::Manager and
  /// Propagator::printBreakdowns() to identify the parameters of a dial.
  /// @param iInput_ Input position (0-based, order of addParameterReference()).
  /// @return The parameter. No bounds check.
  /// @pre setParSetRef() has been called.
  [[nodiscard]] auto& getParameter(int iInput_) const { return _inputParameterReferenceList_[iInput_].getParameter(_parSetListPtr_); }
  /// Gives the mirror window of one input. Cache::Manager reads it to set up
  /// the same mirroring on the GPU.
  /// @param iInput_ Input position (0-based, order of addParameterReference()).
  /// @return The mirror edges; NaN members mean no mirroring. No bounds check.
  [[nodiscard]] auto& getMirrorEdges(int iInput_) const{ return _inputParameterReferenceList_[iInput_].mirrorEdges; }

  /// Push the index of a ParameterSet and Parameter in the set onto the
  /// vector of parameters.  This must be used in the order that the dial will
  /// be expecting the parameters (e.g. for a 2D parameter,
  ///
  /// @note The sentence above is cut off. What it means: input `i` of the
  ///       dial reads the parameter of the `i`-th reference added, for
  ///       example the first and second coordinates of a 2D dial such as
  ///       Bilinear.
  ///
  /// Called by DialCollection::configureImpl() (from the `dialInputList`
  /// option) and DialCollection::setupDialInterfaceReferences().
  /// @param parReference_ Reference to copy. Its `bufferIndex` is ignored:
  ///        the copy gets the next free slot.
  /// @pre initialise() has not been called; otherwise the job stops with an
  ///      error.
  void addParameterReference( const ParameterReference& parReference_);

  /// Simple printout function for debug info on error
  /// @param shallow_ Unused.
  /// @return One item per input: `FULL_TITLE=VALUE` with the current
  ///         parameter value (before mirroring), or `{SET_INDEX, PAR_INDEX}`
  ///         if no parameter-set list is set; followed by the mirror edges
  ///         when mirroring is on.
  [[nodiscard]] std::string getSummary(bool shallow_ = true) const;

private:
  /// Flag if the member can be still edited.
  /// Set to true by initialise(); once true, addParameterReference() throws.
  bool _isInitialized_{false};

  /// Flag for if the dials need to be recalculated.  Dials don't need to
  /// be recalculated if the parameter values have not changed.
  /// Starts true so that the first evaluation always happens; rewritten by
  /// every update().
  bool _isDialUpdateRequested_{true};
  /// Monotonic invalidation token, independent of how often update() is called.
  /// Increased by update() when an input changed, by invalidateBuffers() and
  /// by setInputValues(); see getRevision().
  std::uint64_t _revision_{0};

  /// How many inputs are handled
  ///
  /// Set by initialise(); 0 before.
  int _inputArraySize_{0};

  /// A pointer to the "global" vector of parameter sets.
  /// This provides the connection to the parameters.
  /// Non-owning; set by setParSetRef().
  std::vector<ParameterSet>* _parSetListPtr_{nullptr};

  /// A calculated buffer of parameter values to be handed to the dial
  /// evaluation.  This is the value after transformations like "mirroring"
  /// have been applied.  For a 1D dial, this has a size of 1.  For a
  /// 2D this has a size of 2, and so on.
  /// Sized by initialise(). NaN means not filled yet, or invalidated by
  /// invalidateBuffers().
  std::vector<double> _inputBuffer_;

  /// References to the parameters this buffer reads, in input order.
  ///
  /// There is one entry per parameter used by the dial (e.g. a 1D parameter
  /// has one entry)
  std::vector<ParameterReference> _inputParameterReferenceList_{};

};


#endif //GUNDAM_DIAL_INPUT_BUFFER_H
