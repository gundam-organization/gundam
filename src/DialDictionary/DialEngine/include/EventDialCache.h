//
// Created by Adrien Blanchet on 01/12/2022.
//

#ifndef GUNDAM_EVENT_DIAL_CACHE_H
#define GUNDAM_EVENT_DIAL_CACHE_H

#include "SampleSet.h"
#include "DialCollection.h"
#include "Event.h"
#include "DialInterface.h"

#include "GundamGlobals.h"

#include "GenericToolbox.Wrappers.h"

#include <vector>
#include <utility>


class EventDialCache{

public:

  /// GlobalEventReweightCap is a struct keeping the properties of the reweight capping
  struct GlobalEventReweightCap{
    bool isEnabled{false};
    double maxReweight{std::nan("unset")};

    /// apply the cap if enabled
    void process( double& reweightValue_ ) const{
      if( not isEnabled ){ return; }
      if( reweightValue_ > maxReweight ){ reweightValue_ = maxReweight; }
    }
  };

  /// DialResponseCache is keeping a reference of a DialInterface and a cached double for the response
  struct DialResponseCache {
    DialResponseCache() = delete; // prevent not setting up the interface ptr
    explicit DialResponseCache( DialInterface& interface_ )
      : dialInterface(&interface_) {
      this->updateRequested = dialInterface->getInputBufferRef()->isDialUpdateRequestedPtr();
    }
    // The dial interface to be used with the Event.
    DialInterface* dialInterface{nullptr};
    // The cached result calculated by the dial.
    double response{std::nan("unset")};
    // A cached boolean to check if the dial needs to be updated.
    bool *updateRequested{nullptr};
    void update(){
      // Reevaluate the dial if an update has been requested
      if( *(this->updateRequested) ) {
        response = dialInterface->evalResponse();
      }
    }
    double getResponse(){
      this->update();
      return response;
    }
  };

  /// The cache element associating a Event to the appropriate
  /// DialInterface.
  struct CacheEntry {
    Event* event;
    std::vector<DialResponseCache> dialResponseCacheList{};

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
  };

  /// A pair of indices into the vector of dial collections, and then the
  /// index of the dial interfaces in the dial collection vector of dial
  /// interfaces.
  struct DialIndexCacheEntry {
    /// The index in the dial collection being associated with the event.
    std::size_t collectionIndex {std::size_t(-1)};
    /// The index of the actual dial interface in the dial collection being
    /// associated with the event.
    std::size_t interfaceIndex {std::size_t(-1)};

    [[nodiscard]] std::string getSummary(bool shallow_ = true) const{
      std::stringstream ss;
      ss << "collection: " << collectionIndex << ", interface: " << interfaceIndex;
      return ss.str();
    }
    friend std::ostream& operator <<( std::ostream& o, const DialIndexCacheEntry& this_ ){
      o << this_.getSummary(); return o;
    }
  };

  /// A pair of indices into the the vector of Samples in the SampleSet
  /// vector of fit samples, and the index of the event eventList in the
  /// SampleElement returned by getMcContainer().
  struct EventIndexCacheEntry {
    /// The index of the fit sample being referenced by this indexed cache
    /// entry in the SampleSet vector of Sample objects (returned by
    /// getSampleList().
    std::size_t sampleIndex {std::size_t(-1)};
    /// The index of the MC event being reference by this indexed cache entry
    /// in the SampleElement eveltList vector for the SampleElement returned
    /// by Sample::getMcContainer()
    std::size_t eventIndex {std::size_t(-1)};

    [[nodiscard]] std::string getSummary(bool shallow_ = true) const{
      std::stringstream ss;
      ss << "sample: " << sampleIndex << ", idx: " << eventIndex;
      return ss.str();
    }
    friend std::ostream& operator <<( std::ostream& o, const EventIndexCacheEntry& this_ ){
      o << this_.getSummary(); return o;
    }
  };

  /// A mapping between the event (in the SampleSet, and the dial (in the
  /// DialCollectionVector).  This will be used to build a fast lookup table
  /// between the PhysicsEvent* and the DialInterface* (i.e. a CacheElem_t
  /// returned by getCache().
  struct IndexedCacheEntry{
    EventIndexCacheEntry event;
    std::vector<DialIndexCacheEntry> dials;

    [[nodiscard]] std::string getSummary(bool shallow_ = true) const{
      std::stringstream ss;
      ss << "Event{" << event << "}";
      ss << " -> Dials" << GenericToolbox::toString(dials);
      return ss.str();
    }
    friend std::ostream& operator <<( std::ostream& o, const IndexedCacheEntry& this_ ){
      o << this_.getSummary(); return o;
    }
  };

public:
  EventDialCache() = default;


  // returns the current index
  [[nodiscard]] size_t getFillIndex() const { return _fillIndex_; }

  [[nodiscard]] const std::vector<IndexedCacheEntry>& getIndexedCache() const { return _indexedCache_; }

  /// Provide the event dial cache.  The event dial cache containes a
  /// CacheElem_t object for every dial applied to a physics event.  The
  /// CacheElem_t is a pointer to the PhysicsEvent that will be reweighted and
  /// a pointer to the DialInterface that will do the reweighting.  Note: Each
  /// PhysicsEvent will probably be in the cache multiple times (for different
  /// DialInterface objects), and each DialInterface object could be in the
  /// cache multiple times (but for different Physics event objects).
  std::vector<CacheEntry> &getCache(){ return _cache_; }
  [[nodiscard]] const std::vector<CacheEntry> &getCache() const{ return _cache_; }

  GlobalEventReweightCap& getGlobalEventReweightCap(){ return _globalEventReweightCap_; }

  /// Allocate entries for events in the indexed cache.  The first parameter
  /// arethe number of events to allocate space for, and the second number is
  /// the total number of dials that might exist for each event.
  void allocateCacheEntries(size_t nEvent_, size_t nDialsMaxPerEvent_);

  /// This gets the next available indexed cache entry.  WARNING: This is a
  /// bare pointer but it is referencing an element of a vector and can be
  /// invalidated if the values get added to the indexed cache.  The ownership
  /// of the pointer is not passed to the caller.
  IndexedCacheEntry* fetchNextCacheEntry();

  /// Build the association between pointers to PhysicsEvent objects and the
  /// pointers to DialInterface objects.  This must be done before the event
  /// dial cache can be used, but after the index cache has been filled.
  void buildReferenceCache(SampleSet& sampleSet_,
                           std::vector<DialCollection>& dialCollectionList_);

  /// Resize the cache vectors to remove entries with null events
  void shrinkIndexedCache();

  // copy from
  void fillCacheEntries(const SampleSet& sampleSet_);

  void reweightEntry( CacheEntry& entry_);


private:
  // The next available entry in the indexed cache.
  size_t _fillIndex_{0};

  // A cache mapping events to dials.  This is built while the dials are
  // allocated, and might contain "empty" or invalid entries since some events
  // can have dials that get skipped.  The indexedCache will be used to build
  // the main vector of CacheElem_t which will only have valid pairs of events
  // and dials.
  std::vector<IndexedCacheEntry> _indexedCache_{};

  /// A cache of all of the valid PhysicsEvent* and DialInterface*
  /// associations for efficient use when reweighting the MC events.
  std::vector<CacheEntry> _cache_{};

  /// Global cap
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
