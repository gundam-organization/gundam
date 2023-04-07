//
// Created by Adrien Blanchet on 01/12/2022.
//

#ifndef GUNDAM_EVENTDIALCACHE_H
#define GUNDAM_EVENTDIALCACHE_H

#include "FitSampleSet.h"
#include "DialCollection.h"
#include "PhysicsEvent.h"
#include "DialInterface.h"


// DEV
#include "GlobalVariables.h"

#include "GenericToolbox.Wrappers.h"

#include "vector"
#include "utility"

class EventDialCache {

public:
  EventDialCache() = default;

#ifndef USE_BREAKDOWN_CACHE
  // The dial interface to be used with the PhysicsEvent
  typedef DialInterface* DialsElem_t;
#else
  struct DialsElem_t {
    DialsElem_t(DialInterface* d,double w): dial(d), result(w) {}
    // The dial interface to be used with the PhysicsEvent.
    DialInterface* dial;
    // The cached result calculated by the dial.
    double result;
  };
#endif

  /// The cache element associating a PhysicsEvent to the appropriate
  /// DialInterface.
  struct CacheElem_t {
     PhysicsEvent* event;
     std::vector<DialsElem_t> dials;
  };

  typedef std::pair<size_t, size_t> EventIndexEntry_t;
  typedef std::pair<size_t, size_t> DialIndexEntry_t;
  typedef std::pair<EventIndexEntry_t, std::vector<DialIndexEntry_t>> IndexedEntry_t;

  /// Provide the event dial cache.  The event dial cache containes a
  /// CacheElem_t object for every dial applied to a physics event.  The
  /// CacheElem_t is a pointer to the PhysicsEvent that will be reweighted and
  /// a pointer to the DialInterface that will do the reweighting.  Note: Each
  /// PhysicsEvent will probably be in the cache multiple times (for different
  /// DialInterface objects), and each DialInterface object could be in the
  /// cache multiple times (but for different Physics event objects).
  std::vector<CacheElem_t> &getCache();

  /// Allocate entries for events in the indexed cache.  The first parameter
  /// arethe number of events to allocate space for, and the second number is
  /// the total number of dials that might exist for each event.
  void allocateCacheEntries(size_t nEvent_, size_t nDialsMaxPerEvent_);

  /// This gets the next available indexed cache entry.  WARNING: This is a
  /// bare pointer but it is referencing an element of a vector and can be
  /// invalidated if the values get added to the indexed cache.
  IndexedEntry_t* fetchNextCacheEntry();

  /// Build the association between pointers to PhysicsEvent objects and the
  /// pointers to DialInterface objects.  This must be done before the event
  /// dial cache can be used, but after the index cache has been filled.
  void buildReferenceCache(FitSampleSet& sampleSet_,
                           std::vector<DialCollection>& dialCollectionList_);

  static void reweightEntry(CacheElem_t& entry_);

private:
  // The next available entry in the indexed cache.
  size_t _fillIndex_{0};

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

  /// Flag that all events should have dials (or not).
  bool _warnForDialLessEvent_{false};


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
