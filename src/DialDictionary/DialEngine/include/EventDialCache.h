//
// Created by Adrien Blanchet on 01/12/2022.
//

/// @file EventDialCache.h
/// @brief Declares EventDialCache, the table that links each event to the
///        dials that weight it and recomputes event weights from it.
/// @ingroup DialDictionary

#ifndef GUNDAM_EVENT_DIAL_CACHE_H
#define GUNDAM_EVENT_DIAL_CACHE_H

#include "SampleSet.h"
#include "DialCollection.h"
#include "Event.h"
#include "DialInterface.h"

#include "GundamGlobals.h"

#include "GenericToolbox.Wrappers.h"

#include <memory>
#include <cstdint>
#include <vector>
#include <utility>


/// Links every event of a SampleSet to the dials that weight it, and
/// recomputes the event weights from the dial responses.
///
/// A dial is a response function of one or more fit parameters
/// (DialInterface, owned by a DialCollection). EventDialCache is the join
/// table between the events of a Propagator's SampleSet and the dials of
/// that Propagator's DialManager. It is built in two stages:
/// 1. **Indexed cache** (IndexedCacheEntry), filled while the data is
///    loaded. DataDispenser reserves entries with allocateCacheEntries(),
///    claims one per event with fetchNextCacheEntry(), and writes into it the
///    position of the event (sample index, event index) and of each dial that
///    applies to it (collection index, interface index). Indices are used
///    because the event and dial containers are still being filled and may
///    still be reallocated (DialManager::shrinkDialContainers()).
/// 2. **Reference cache** (CacheEntry), built by
///    Propagator::buildDialCache() with shrinkIndexedCache() and
///    buildReferenceCache() once the containers are final: one entry per
///    event, holding a pointer to the Event and one DialResponseCache (a
///    DialInterface pointer plus the last response) per dial. Dials of
///    composite-node collections are not listed on their own: they are
///    reached through their parent CompositeDial. Composite and
///    event-formula dials get an event-specific copy (see
///    CacheEntry::boundDialInterfaces).
///
/// For every likelihood evaluation, Propagator::reweightEvents() calls
/// reweightEntry() on each CacheEntry, splitting getCache() between threads.
/// Dials whose collection sets `applyToVariable` change an event variable
/// instead of the weight; EventVariableResponseCache applies them with
/// applyVariableResponses().
/// The GPU Cache::Manager reads the same reference cache (getCache()) and the
/// global event-weight cap (getGlobalEventReweightCap()) when it builds its
/// own copy of the calculation.
///
/// @par Ownership
/// Each Propagator owns one EventDialCache by value. The cache owns its
/// entries. Events belong to the samples of the SampleSet and DialInterface
/// objects to their DialCollection; the reference cache only holds raw
/// pointers to them. The exception is the event-specific interfaces of
/// composite and event-formula dials, which each CacheEntry owns
/// (CacheEntry::boundDialInterfaces).
///
/// @par Life cycle
/// 1. Propagator::clearContent() replaces it with a new, empty cache that
///    keeps the global event-weight cap.
/// 2. Data loading: DataDispenser calls allocateCacheEntries() and
///    fetchNextCacheEntry(). When events are copied from another propagator
///    instead (Propagator::copyEventsFrom()), fillCacheEntries() creates one
///    entry per event, with no dial.
/// 3. Propagator::buildDialCache(): shrinkIndexedCache(), then
///    buildReferenceCache(), which empties the indexed cache.
/// 4. Every reweight: reweightEntry() for each CacheEntry.
///
/// @par Thread safety
/// fetchNextCacheEntry() is not thread-safe; DataDispenser calls it while
/// holding its own mutex. reweightEntry() can run on different CacheEntry
/// objects at the same time, but not while DialManager::updateDialState()
/// updates the dial input buffers.
///
/// @warning The reference cache holds raw pointers into the samples' event
///          lists and the collections' DialInterface lists. They dangle if
///          those vectors are reallocated after buildReferenceCache().
/// @see DialManager, DialCollection, DialInterface, DialInputBuffer, Propagator
/// @ingroup DialDictionary
class EventDialCache{

public:

  /// GlobalEventReweightCap is a struct keeping the properties of the reweight capping
  ///
  /// The cap is an upper limit on the product of the dial responses of one
  /// event (its reweight factor), applied before that product multiplies the
  /// event's base weight. Propagator::configureImpl() reads `maxReweight`
  /// from the Propagator option `globalEventReweightCap`, and
  /// Propagator::initializeImpl() sets `isEnabled` when that value is not
  /// NaN. Cache::Manager passes the same cap to the GPU.
  /// Propagator::clearContent() replaces the whole EventDialCache, but copies
  /// these settings into the new one.
  struct GlobalEventReweightCap{
    bool isEnabled{false}; ///< Whether process() applies the cap; false by default.
    double maxReweight{std::nan("unset")}; ///< Largest allowed reweight factor; NaN if unset.

    /// apply the cap if enabled
    ///
    /// Lowers the value to `maxReweight` if it is larger. There is no lower
    /// cap.
    /// @param reweightValue_ Reweight factor of one event, modified in place.
    void process( double& reweightValue_ ) const{
      if( not isEnabled ){ return; }
      if( reweightValue_ > maxReweight ){ reweightValue_ = maxReweight; }
    }
  };

  /// DialResponseCache is keeping a reference of a DialInterface and a cached double for the response
  ///
  /// There is one per (event, dial) link, inside a CacheEntry. It evaluates
  /// the dial again only when the revision of the dial's DialInputBuffer
  /// changed since the last evaluation (an input changed, or the buffer was
  /// invalidated), and otherwise returns the response computed last time.
  /// Comparing revisions rather than reading the "update requested" flag
  /// means a change is not missed when the buffer is updated again before
  /// this response is read. Each event keeps its own cached value, even when
  /// several events share the same dial.
  struct DialResponseCache {
    /// Not available: a cache entry must be built from a DialInterface.
    DialResponseCache() = delete; // prevent not setting up the interface ptr
    /// Links the cache to a dial.
    /// @param interface_ Dial to use. It must outlive this object and must
    ///        already have its input buffer set: the constructor reads the
    ///        address of the buffer's revision counter from it.
    /// @note `response` starts as NaN and `inputRevision` at a value no
    ///       buffer has, so the first getResponse() computes the response.
    explicit DialResponseCache( DialInterface& interface_ )
      : dialInterface(&interface_), inputRevisionPtr(interface_.getInputBufferRef()->getRevisionPtr()) {}
    /// The dial interface to be used with the Event.
    /// Non-owning; points into a DialCollection's DialInterface list, or to
    /// an event-specific interface kept in CacheEntry::boundDialInterfaces.
    DialInterface* dialInterface{nullptr};
    /// True when the dial changes an event variable (the collection's
    /// `applyToVariable` option) instead of the event weight: reweightEntry()
    /// skips it and applyVariableResponses() uses it.
    bool appliesToVariable{false};
    /// The cached result calculated by the dial.
    /// Already clamped by the dial's DialResponseSupervisor; NaN until first
    /// computed.
    double response{std::nan("unset")};
    /// Address of the revision counter of the dial's input buffer
    /// (DialInputBuffer::getRevisionPtr()).
    ///
    /// Cache the revision address to avoid following the interface and input
    /// buffer pointers for every response, including unchanged parameters.
    const std::uint64_t* inputRevisionPtr{nullptr};
    /// Input-buffer revision for which `response` was last computed. Starts at
    /// the largest value, which no buffer reaches, so the first update()
    /// always evaluates the dial.
    std::uint64_t inputRevision{std::uint64_t(-1)};
    /// Recomputes `response` with DialInterface::evalResponse() if the input
    /// buffer's revision changed since the last evaluation; otherwise does
    /// nothing.
    void update(){
      // Retain changes until this response consumes them, even when the
      // input buffer has been updated again in the meantime.
      auto revision = *inputRevisionPtr;
      if( inputRevision != revision ) {
        response = dialInterface->evalResponse();
        inputRevision = revision;
      }
    }
    /// Gives the dial response, recomputing it first if needed (update()).
    /// @return The current, clamped response of the dial.
    double getResponse(){
      this->update();
      return response;
    }
  };

  /// Per-event target, with responses indexed into dialResponseCacheList.
  /// The loaded reference is captured after sorting events, before propagation.
  ///
  /// One per event variable that variable dials act on (collections with
  /// `applyToVariable`), filled by buildReferenceCache() and used by
  /// applyVariableResponses(). The collection's `variableResponseMode`
  /// decides the list a dial goes to.
  struct VariableResponseCache {
    int variableIndex{-1}; ///< Position of the variable in the event's variable list.
    double referenceValue{0}; ///< Value of the variable when the cache was built.
    std::vector<std::size_t> multiplyDials; ///< Dials whose response multiplies the value (`Multiply`).
    std::vector<std::size_t> addDials; ///< Dials whose response is added to the value (`Add`).
    std::size_t overrideDial{std::size_t(-1)}; ///< Dial whose response replaces the reference value (`Override`); `size_t(-1)` if none. At most one per variable.
  };

  /// The cache element associating a Event to the appropriate
  /// DialInterface.
  ///
  /// There is one per event in the reference cache (getCache()): the event
  /// to reweight, and one DialResponseCache per dial that applies to it. The
  /// dial list can be empty.
  struct CacheEntry {
    /// Event to reweight, owned by its Sample.
    /// Has no default member initializer; buildReferenceCache() creates
    /// entries with `emplace_back()`, which value-initializes it to nullptr,
    /// then sets it.
    Event* event;
    /// One cached dial link per dial that weights this event; empty if no
    /// dial applies.
    std::vector<DialResponseCache> dialResponseCacheList{};
    /// One target per event variable changed by variable dials; empty if
    /// no such dial applies.
    std::vector<VariableResponseCache> variableResponseCacheList{};
    /// Stable ownership for event-specific formulas and composites. Ordinary interfaces
    /// continue to belong to their DialCollection.
    ///
    /// buildReferenceCache() creates these interfaces for EventFormula dials
    /// (which read event variables) and for CompositeDial copies bound to
    /// this event's node dials. The DialResponseCache entries point to them.
    std::vector<std::shared_ptr<DialInterface>> boundDialInterfaces{};

    /// Describes the event and its dials for debug printouts. Used by
    /// Propagator::printBreakdowns() and
    /// LikelihoodInterface::throwToyParameters() when the
    /// `debugPrintLoadedEvents` option is set.
    /// @param shallow_ Passed to DialInterface::getSummary() for each dial.
    /// @return The event printout, then one line per dial inside `Dials{...}`;
    ///         a fixed message if `event` is null.
    /// @note Each DialInterface::getSummary() call evaluates its dial again.
    [[nodiscard]] std::string getSummary(bool shallow_ = true) const {
      std::stringstream ss;
      if( event == nullptr ){ return {"No event attached to this cache entry."}; }
      ss << *event;
      if( not dialResponseCacheList.empty() ){
        ss << std::endl << "Dials{";
        for( auto& dialResponseCache : dialResponseCacheList ){
          ss << std::endl << "  { " << dialResponseCache.dialInterface->getSummary(shallow_) << " }";
        }
        ss << std::endl << "}";
      }
      return ss.str();
    }
    /// Tells whether a parameter is an input of any dial of this event.
    /// @param parameter_ Parameter to look for, compared by address.
    /// @return True if getDialIndicesAffecting() is not empty.
    /// @note Not called in the current code.
    [[nodiscard]] bool isEventAffectedBy(const Parameter& parameter_) {
      return not getDialIndicesAffecting(parameter_).empty();
    }
    /// Lists the dials of this event that read a given parameter. Exposed to
    /// the Python interface.
    /// @param parameter_ Parameter to look for, compared by address with the
    ///        inputs of each dial's DialInputBuffer.
    /// @return Positions in `dialResponseCacheList`; a dial appears once per
    ///         input that is this parameter.
    [[nodiscard]] std::vector<int> getDialIndicesAffecting(const Parameter& parameter_) {
      std::vector<int> out;
      // loop over the dials attached to this event
      int idx{-1};
      for( auto& dialCache : dialResponseCacheList ) {
        idx++;
        // some dials may have > 1 paramater as input
        for( size_t inputIndex = 0 ; inputIndex < dialCache.dialInterface->getInputBufferRef()->getInputSize() ; inputIndex++ ){
          if( &dialCache.dialInterface->getInputBufferRef()->getParameter(int(inputIndex)) == &parameter_ ) {
            out.emplace_back(idx);
          }
        }
      }
      return out;
    }
  };

  /// A pair of indices into the vector of dial collections, and then the
  /// index of the dial interfaces in the dial collection vector of dial
  /// interfaces.
  ///
  /// Both indices are `size_t(-1)` for an unused dial slot.
  struct DialIndexCacheEntry {
    /// The index in the dial collection being associated with the event.
    /// It is the position of the collection in
    /// DialManager::getDialCollectionList() (DialCollection::getIndex());
    /// `size_t(-1)` when unused.
    std::size_t collectionIndex {std::size_t(-1)};
    /// The index of the actual dial interface in the dial collection being
    /// associated with the event.
    /// It is a position in DialCollection::getDialInterfaceList();
    /// `size_t(-1)` when unused.
    std::size_t interfaceIndex {std::size_t(-1)};

    /// Describes the slot for debug printouts.
    /// @param shallow_ Unused.
    /// @return `{ collection: C, interface: I}`, or an empty string when both
    ///         indices are unset.
    [[nodiscard]] std::string getSummary(bool shallow_ = true) const{
      if( collectionIndex == std::size_t(-1) and interfaceIndex == std::size_t(-1) ){ return {}; }
      std::stringstream ss;
      ss << "{ collection: " << collectionIndex << ", interface: " << interfaceIndex << "}";
      return ss.str();
    }
    /// Writes getSummary() to a stream.
    /// @param o Output stream.
    /// @param this_ Slot to print.
    /// @return `o`.
    friend std::ostream& operator <<( std::ostream& o, const DialIndexCacheEntry& this_ ){
      o << this_.getSummary(); return o;
    }
  };

  /// A pair of indices into the the vector of Samples in the SampleSet
  /// vector of fit samples, and the index of the event eventList in the
  /// SampleElement returned by getMcContainer().
  ///
  /// @note SampleElement and getMcContainer() no longer exist: `eventIndex`
  ///       is a position in Sample::getEventList().
  struct EventIndexCacheEntry {
    /// The index of the fit sample being referenced by this indexed cache
    /// entry in the SampleSet vector of Sample objects (returned by
    /// getSampleList().
    /// `size_t(-1)` until set.
    std::size_t sampleIndex {std::size_t(-1)};
    /// The index of the MC event being reference by this indexed cache entry
    /// in the SampleElement eveltList vector for the SampleElement returned
    /// by Sample::getMcContainer()
    ///
    /// @note Now a position in Sample::getEventList(); `size_t(-1)` until
    ///       set.
    std::size_t eventIndex {std::size_t(-1)};

    /// Describes the event position for debug printouts.
    /// @param shallow_ Unused.
    /// @return `sample: S, idx: E`.
    [[nodiscard]] std::string getSummary(bool shallow_ = true) const{
      std::stringstream ss;
      ss << "sample: " << sampleIndex << ", idx: " << eventIndex;
      return ss.str();
    }
    /// Writes getSummary() to a stream.
    /// @param o Output stream.
    /// @param this_ Entry to print.
    /// @return `o`.
    friend std::ostream& operator <<( std::ostream& o, const EventIndexCacheEntry& this_ ){
      o << this_.getSummary(); return o;
    }
  };

  /// A mapping between the event (in the SampleSet, and the dial (in the
  /// DialCollectionVector).  This will be used to build a fast lookup table
  /// between the PhysicsEvent* and the DialInterface* (i.e. a CacheElem_t
  /// returned by getCache().
  ///
  /// @note PhysicsEvent and CacheElem_t are now Event and CacheEntry.
  struct IndexedCacheEntry{
    /// Position of the event in the SampleSet.
    EventIndexCacheEntry event;
    /// One slot per dial that may apply to the event. allocateCacheEntries()
    /// sizes it with unused slots; DataDispenser fills them from the front,
    /// at most one per dial collection. Unused slots are skipped later.
    std::vector<DialIndexCacheEntry> dials;

    /// Describes the entry for debug printouts.
    /// @param shallow_ Unused.
    /// @return The event position followed by the list of dial slots.
    [[nodiscard]] std::string getSummary(bool shallow_ = true) const{
      std::stringstream ss;
      ss << "Event{" << event << "}";
      ss << " -> Dials" << GenericToolbox::toString(dials);
      return ss.str();
    }
    /// Writes getSummary() to a stream.
    /// @param o Output stream.
    /// @param this_ Entry to print.
    /// @return `o`.
    friend std::ostream& operator <<( std::ostream& o, const IndexedCacheEntry& this_ ){
      o << this_.getSummary(); return o;
    }
  };

public:
  /// Creates an empty cache, with the global event-weight cap disabled.
  EventDialCache() = default;


  // returns the current index
  /// Gives the number of indexed-cache entries handed out so far.
  ///
  /// DataDispenser compares it with its `debugNbMaxEventsToLoad` option to
  /// stop loading early.
  /// @return Index of the entry the next fetchNextCacheEntry() call returns.
  [[nodiscard]] size_t getFillIndex() const { return _fillIndex_; }

  /// Gives read-only access to the indexed cache.
  /// @return The indexed cache; empty after buildReferenceCache().
  /// @note Not called in the current code.
  [[nodiscard]] auto& getIndexedCache() const { return _indexedCache_; }

  /// Provide the event dial cache.  The event dial cache containes a
  /// CacheElem_t object for every dial applied to a physics event.  The
  /// CacheElem_t is a pointer to the PhysicsEvent that will be reweighted and
  /// a pointer to the DialInterface that will do the reweighting.  Note: Each
  /// PhysicsEvent will probably be in the cache multiple times (for different
  /// DialInterface objects), and each DialInterface object could be in the
  /// cache multiple times (but for different Physics event objects).
  /// @return The reference cache.
  /// @note Outdated: the cache now holds one CacheEntry per event, which
  ///       lists all the dials of that event, so each Event appears exactly
  ///       once. A DialInterface can still appear in many entries.
  ///       CacheElem_t and PhysicsEvent are now CacheEntry and Event.
  std::vector<CacheEntry> &getCache(){ return _cache_; }
  /// Gives read-only access to the reference cache.
  /// @return One CacheEntry per event; empty until buildReferenceCache().
  [[nodiscard]] const std::vector<CacheEntry> &getCache() const{ return _cache_; }

  /// Allocated cache vector buffers, excluding events, dial objects and allocator overhead.
  ///
  /// Counts the capacity of the indexed and reference caches and of each
  /// entry's dial list, but not the variable-response lists or the bound
  /// interfaces. Logged by buildReferenceCache() and exposed to the Python
  /// interface.
  /// @return Size in bytes.
  [[nodiscard]] size_t getCacheMemorySize() const;

  /// Gives mutable access to the global event-weight cap. Propagator sets it
  /// up during configuration and initialization; Cache::Manager reads it.
  /// @return The cap settings.

  GlobalEventReweightCap& getGlobalEventReweightCap(){ return _globalEventReweightCap_; }

  /// Allocate entries for events in the indexed cache.  The first parameter
  /// arethe number of events to allocate space for, and the second number is
  /// the total number of dials that might exist for each event.
  ///
  /// The new entries are appended with unset indices; existing entries and
  /// the fill index are kept, so several data sets can be loaded one after
  /// the other.
  /// @param nEvent_ Number of entries to append.
  /// @param nDialsMaxPerEvent_ Number of dial slots in each new entry.
  ///        DataDispenser passes the number of dial collections it loads,
  ///        since an event gets at most one dial per collection, and 0 when
  ///        it fills samples from histogram content.
  /// @warning May reallocate the indexed cache, which invalidates pointers
  ///          returned earlier by fetchNextCacheEntry().
  void allocateCacheEntries(size_t nEvent_, size_t nDialsMaxPerEvent_);

  /// This gets the next available indexed cache entry.  WARNING: This is a
  /// bare pointer but it is referencing an element of a vector and can be
  /// invalidated if the values get added to the indexed cache.  The ownership
  /// of the pointer is not passed to the caller.
  /// @return The entry at the fill index; the fill index is then incremented.
  /// @pre An allocated entry is left; otherwise the job stops with an error.
  /// @note Not thread-safe. DataDispenser calls it while holding its mutex.
  IndexedCacheEntry* fetchNextCacheEntry();

  /// Build the association between pointers to PhysicsEvent objects and the
  /// pointers to DialInterface objects.  This must be done before the event
  /// dial cache can be used, but after the index cache has been filled.
  ///
  /// For each sample, it:
  /// 1. collects the sample's indexed-cache entries, dropping entries and
  ///    dial slots with unset indices;
  /// 2. sorts the sample's events by TChain entry number
  ///    (`Event::getIndices().entry`) and applies the same reordering to
  ///    the collected entries;
  /// 3. appends one CacheEntry per event to the reference cache, with one
  ///    DialResponseCache per valid dial slot.
  ///
  /// Called by Propagator::buildDialCache().
  /// @param sampleSet_ Samples whose events are linked. Their event lists
  ///        are reordered.
  /// @param dialCollectionList_ The collections the dial indices refer to
  ///        (DialManager::getDialCollectionList()).
  /// @pre Every event of every sample has exactly one indexed-cache entry,
  ///      otherwise the job stops with an error. The entries of a sample
  ///      must also appear in the same order as its events: they are matched
  ///      by position (DataDispenser claims an entry and an event slot
  ///      together, under one lock).
  /// @pre DialManager::shrinkDialContainers() has run, so the DialInterface
  ///      objects do not move any more.
  /// @post The indexed cache is empty. The new entries are appended to the
  ///       reference cache, which is not cleared first.
  /// @warning Side effect: reorders Sample::getEventList() of every sample.
  /// @note PhysicsEvent is now Event.
  void buildReferenceCache(SampleSet& sampleSet_, std::vector<DialCollection>& dialCollectionList_);

  /// Resize the cache vectors to remove entries with null events
  ///
  /// Truncates the indexed cache to `getFillIndex() + 1` entries and asks
  /// the vector to release the unused memory. Called by
  /// Propagator::buildDialCache().
  /// @note Only the indexed cache is resized, not the reference cache, and
  ///       entries hold indices, not event pointers. One entry more than
  ///       were claimed is kept (or added, with unset indices, if every
  ///       entry was claimed); buildReferenceCache() skips it.
  void shrinkIndexedCache();

  // copy from
  /// Creates one indexed-cache entry with no dial for every event of a
  /// SampleSet, in sample order.
  ///
  /// Called by Propagator::copyEventsFrom() after the events of another
  /// propagator have been copied into this one.
  /// @param sampleSet_ Sample set whose events get an entry.
  /// @note Resets the fill index to 0 but appends the new entries after any
  ///       existing ones; entries past the fill index stay until
  ///       shrinkIndexedCache().
  /// @note The entries carry no dial. If Propagator::buildDialCache() and a
  ///       reweight follow (LikelihoodInterface::loadDataPropagator() does
  ///       this for data sets with `evalModelAt`), reweightEntry() sets each
  ///       copied event's current weight back to its base weight. In the
  ///       Asimov case buildDialCache() is not called, so the reference cache
  ///       stays empty.
  void fillCacheEntries(const SampleSet& sampleSet_);

  /// Sets the event variables that variable dials act on.
  ///
  /// For each target variable of the entry (VariableResponseCache), starts
  /// from the value the variable had when the cache was built, or from the
  /// response of its `Override` dial if it has one, multiplies it by the
  /// responses of its `Multiply` dials, adds those of its `Add` dials, and
  /// stores the result in the event. EventVariableResponseCache::update()
  /// calls it from several threads, each on its own range of getCache().
  /// @param entry_ Entry whose event variables are updated.
  void applyVariableResponses(CacheEntry& entry_);
  /// Tells whether any dial applies to an event variable.
  /// @return True if buildReferenceCache() linked at least one dial whose
  ///         collection sets `applyToVariable`.
  [[nodiscard]] bool hasVariableResponses() const{ return _hasVariableResponses_; }

  /// Recomputes the current weight of one event from its dials.
  ///
  /// Multiplies the responses of the event's dials (each evaluated again
  /// only if its inputs changed, see DialResponseCache::getResponse()),
  /// skipping dials that act on an event variable
  /// (DialResponseCache::appliesToVariable), applies the global event-weight
  /// cap to that product, then sets the event's current weight to its base
  /// weight times the product.
  /// @param entry_ Entry of the event to reweight; `entry_.event` must not be
  ///        null.
  /// @pre DialManager::updateDialState() has run since the last parameter
  ///      change, so the dial input buffers and their flags are current.
  /// @par Thread safety
  /// Propagator::reweightEvents() calls it from several threads, each on its
  /// own range of getCache(). Safe as long as no two threads process the same
  /// entry at the same time.

  void reweightEntry( CacheEntry& entry_);


private:
  /// Set to true by buildReferenceCache() when a dial applies to an event
  /// variable; see hasVariableResponses().
  bool _hasVariableResponses_{false};
  /// The next available entry in the indexed cache.
  /// Incremented by fetchNextCacheEntry(); reset by fillCacheEntries().
  size_t _fillIndex_{0};

  /// A cache mapping events to dials.  This is built while the dials are
  /// allocated, and might contain "empty" or invalid entries since some events
  /// can have dials that get skipped.  The indexedCache will be used to build
  /// the main vector of CacheElem_t which will only have valid pairs of events
  /// and dials.
  ///
  /// @note CacheElem_t is now CacheEntry. buildReferenceCache() clears this
  ///       vector once the reference cache is built.
  std::vector<IndexedCacheEntry> _indexedCache_{};

  /// A cache of all of the valid PhysicsEvent* and DialInterface*
  /// associations for efficient use when reweighting the MC events.
  /// One CacheEntry per event (PhysicsEvent is now Event), built by
  /// buildReferenceCache().
  std::vector<CacheEntry> _cache_{};

  /// Global cap
  ///
  /// Applied by reweightEntry(); see GlobalEventReweightCap.
  GlobalEventReweightCap _globalEventReweightCap_{};
};


#endif //GUNDAM_EVENT_DIAL_CACHE_H

//  A Lesser GNU Public License

//  Copyright (C) 2023 GUNDAM DEVELOPERS

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA

// Local Variables:
// mode:c++
// c-basic-offset:2
// End:
