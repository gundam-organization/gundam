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
  struct CacheElem_t {
     PhysicsEvent* first;
     std::vector<DialInterface*> second;
  };
#else
  struct DialsElem_t {
    DialsElem_t(DialInterface* a,double b): first(a), second(b) {}
    DialInterface* first;
    double second;
  };
  struct CacheElem_t {
     PhysicsEvent* first;
     std::vector<DialsElem_t> second;
  };
#endif

  std::vector<CacheElem_t> &getCache();

  void allocateCacheEntries(size_t nEvent_, size_t nDialsMaxPerEvent_);
  std::pair<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>>* fetchNextCacheEntry();
  void buildReferenceCache(FitSampleSet& sampleSet_, std::vector<DialCollection>& dialCollectionList_);

  static void reweightEntry(CacheElem_t& entry_);

private:
  bool _warnForDialLessEvent_{false};
  size_t _fillIndex_{0};
  GenericToolbox::NoCopyWrapper<std::mutex> _mutex_;
  std::vector<std::pair<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>>> _indexedCache_{};
  std::vector<CacheElem_t> _cache_;

};


#endif //GUNDAM_EVENTDIALCACHE_H
