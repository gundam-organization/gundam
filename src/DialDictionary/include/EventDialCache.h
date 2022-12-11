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
  std::vector<std::pair<PhysicsEvent *, std::vector<DialInterface *>>> &getCache();
#else
  std::vector< std::pair< PhysicsEvent*, std::vector<std::pair<DialInterface*, double> > >> &getCache();
#endif

  void allocateCacheEntries(size_t nEvent_, size_t nDialsMaxPerEvent_);
  std::pair<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>>* fetchNextCacheEntry();
  void buildReferenceCache(FitSampleSet& sampleSet_, std::vector<DialCollection>& dialCollectionList_);

#ifndef USE_BREAKDOWN_CACHE
  static void reweightEntry(std::pair<PhysicsEvent*, std::vector<DialInterface*>>& entry_);
#else
  static void reweightEntry(std::pair<PhysicsEvent*, std::vector<std::pair<DialInterface*, double>>>& entry_);
#endif

private:
  bool _warnForDialLessEvent_{false};
  size_t _fillIndex_{0};
  GenericToolbox::NoCopyWrapper<std::mutex> _mutex_;
  std::vector<std::pair<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>>> _indexedCache_{};
#ifndef USE_BREAKDOWN_CACHE
  std::vector<std::pair<PhysicsEvent*, std::vector<DialInterface*>>> _cache_;
#else
  std::vector< std::pair< PhysicsEvent*, std::vector<std::pair<DialInterface*, double> > > > _cache_;
#endif

};


#endif //GUNDAM_EVENTDIALCACHE_H
