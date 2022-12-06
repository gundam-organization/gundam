//
// Created by Adrien Blanchet on 01/12/2022.
//

#ifndef GUNDAM_EVENTDIALCACHE_H
#define GUNDAM_EVENTDIALCACHE_H

#include "FitSampleSet.h"
#include "DialCollection.h"
#include "PhysicsEvent.h"
#include "DialInterface.h"

#include "GenericToolbox.Wrappers.h"

#include "vector"
#include "utility"

class EventDialCache {

public:
  EventDialCache() = default;

  std::vector<std::pair<PhysicsEvent *, std::vector<DialInterface *>>> &getCache();

  void allocateCacheEntries(size_t nEvent_, size_t nDialsMaxPerEvent_);
  std::pair<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>>* fetchNextCacheEntry();
  void buildReferenceCache(FitSampleSet& sampleSet_, std::vector<DialCollection>& dialCollectionList_);

  static void reweightEntry(std::pair<PhysicsEvent*, std::vector<DialInterface*>>& entry_);

private:
  bool _warnForDialLessEvent_{false};
  size_t _fillIndex_{0};
  GenericToolbox::NoCopyWrapper<std::mutex> _mutex_;
  std::vector<std::pair<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>>> _indexedCache_{};
  std::vector<std::pair<PhysicsEvent*, std::vector<DialInterface*>>> _cache_;

};


#endif //GUNDAM_EVENTDIALCACHE_H
