//
// Created by Adrien Blanchet on 01/12/2022.
//

#ifndef GUNDAM_EVENTDIALCACHE_H
#define GUNDAM_EVENTDIALCACHE_H

#include "SampleSet.h"
#include "DialCollection.h"
#include "PhysicsEvent.h"
#include "DialInterface.h"


// DEV
#include "GundamGlobals.h"

#include "GenericToolbox.Wrappers.h"

#include <vector>
#include <utility>

class EventDialCache {

public:
  static double globalEventReweightCap;

public:
  EventDialCache() = default;

  struct DialsElem_t {
    DialsElem_t(DialInterface* interface_, double response_): interface(interface_), response(response_) {}
    // The dial interface to be used with the PhysicsEvent.
    DialInterface* interface;
    // The cached result calculated by the dial.
    double response;
  };

  /// The cache element associating a PhysicsEvent to the appropriate
  /// DialInterface.
  struct CacheElem_t {
    PhysicsEvent* event;
    std::vector<DialsElem_t> dials;

    std::string getSummary() const {
      std::stringstream ss;
      ss << event->getSummary() << std::endl;
      ss << "dialCache = {";
      for( auto& dialInterface : dials ) {
        ss << std::endl << "  - " << dialInterface.interface->getSummary();
      }
      ss << std::endl << "}";
      return ss.str();
    }
  };

  /// A pair of indices into the vector of dial collections, and then the
  /// index of the dial interfaces in the dial collection vector of dial
  /// interfaces.
  struct DialIndexEntry_t {
    /// The index in the dial collection being associated with the event.
    std::size_t collectionIndex {std::size_t(-1)};
    /// The index of the actual dial interface in the dial collection being
    /// associated with the event.
    std::size_t interfaceIndex {std::size_t(-1)};
  };

  /// A pair of indices into the the vector of FitSamples in the FitSampleSet
  /// vector of fit samples, and the index of the event eventList in the
  /// SampleElement returned by getMcContainer().
  struct EventIndexEntry_t {
    /// The index of the fit sample being referenced by this indexed cache
    /// entry in the FitSampleSet vector of FitSample objects (returned by
    /// getFitSampleList().
    std::size_t sampleIndex {std::size_t(-1)};
    /// The index of the MC event being reference by this indexed cache entry
    /// in the SampleElement eveltList vector for the SampleElement returned
    /// by FitSample::getMcContainer()
    std::size_t eventIndex {std::size_t(-1)};
  };

  /// A mapping between the event (in the FitSampleSet, and the dial (in the
  /// DialCollectionVector).  This will be used to build a fast lookup table
  /// between the PhysicsEvent* and the DialInterface* (i.e. a CacheElem_t
  /// returned by getCache().
  struct IndexedEntry_t{
    EventIndexEntry_t event;
    std::vector<DialIndexEntry_t> dials;
  };

  // returns the current index
  [[nodiscard]] size_t getFillIndex() const { return _fillIndex_; }

  /// Provide the event dial cache.  The event dial cache containes a
  /// CacheElem_t object for every dial applied to a physics event.  The
  /// CacheElem_t is a pointer to the PhysicsEvent that will be reweighted and
  /// a pointer to the DialInterface that will do the reweighting.  Note: Each
  /// PhysicsEvent will probably be in the cache multiple times (for different
  /// DialInterface objects), and each DialInterface object could be in the
  /// cache multiple times (but for different Physics event objects).
  std::vector<CacheElem_t> &getCache(){ return _cache_; }
  [[nodiscard]] const std::vector<CacheElem_t> &getCache() const{ return _cache_; }

  /// Allocate entries for events in the indexed cache.  The first parameter
  /// arethe number of events to allocate space for, and the second number is
  /// the total number of dials that might exist for each event.
  void allocateCacheEntries(size_t nEvent_, size_t nDialsMaxPerEvent_);

  /// This gets the next available indexed cache entry.  WARNING: This is a
  /// bare pointer but it is referencing an element of a vector and can be
  /// invalidated if the values get added to the indexed cache.  The ownership
  /// of the pointer is not passed to the caller.
  IndexedEntry_t* fetchNextCacheEntry();

  /// Build the association between pointers to PhysicsEvent objects and the
  /// pointers to DialInterface objects.  This must be done before the event
  /// dial cache can be used, but after the index cache has been filled.
  void buildReferenceCache(SampleSet& sampleSet_,
                           std::vector<DialCollection>& dialCollectionList_);

  /// Resize the cache vectors to remove entries with null events
  void shrinkIndexedCache();

  static void reweightEntry(CacheElem_t& entry_);


private:
  // The next available entry in the indexed cache.
  size_t _fillIndex_{0};

  // Keep the copy-constructor of the EventDialCache.
  GenericToolbox::NoCopyWrapper<std::mutex> _mutex_;

  // A cache mapping events to dials.  This is built while the dials are
  // allocated, and might contain "empty" or invalid entries since some events
  // can have dials that get skipped.  The indexedCache will be used to build
  // the main vector of CacheElem_t which will only have valid pairs of events
  // and dials.
  std::vector<IndexedEntry_t> _indexedCache_{};

  /// A cache of all of the valid PhysicsEvent* and DialInterface*
  /// associations for efficient use when reweighting the MC events.
  std::vector<CacheElem_t> _cache_;
};


#endif //GUNDAM_EVENTDIALCACHE_H

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
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
