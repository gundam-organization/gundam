//
// Created by Adrien Blanchet on 01/12/2022.
//

#include "EventDialCache.h"

#include "Logger.h"

#ifndef DISABLE_USER_HEADER
LoggerInit([]{
  Logger::setUserHeaderStr("[EventDialCache]");
});
#endif

void EventDialCache::buildReferenceCache( SampleSet& sampleSet_, std::vector<DialCollection>& dialCollectionList_){
  LogInfo << "Building event dial cache..." << std::endl;

  LogInfo << "Indexed cache size: " << _indexedCache_.size() << std::endl;
  LogInfo << "Sorting events in sync with indexed cache..." << std::endl;

  size_t nCacheSlots{0};
  std::vector<std::vector<IndexedCacheEntry>> sampleIndexCacheList{sampleSet_.getSampleList().size()};

  {
    LogScopeIndent;
    LogInfo << "Breaking down indexed cache per sample..." << std::endl;
    for( auto& entry : _indexedCache_ ){
      if( entry.event.sampleIndex == size_t(-1) ){ continue; }
      if( entry.event.eventIndex == size_t(-1) ){ continue; }

      sampleIndexCacheList[entry.event.sampleIndex].emplace_back( entry );
      sampleIndexCacheList[entry.event.sampleIndex].back().dials.clear();
      for( auto& dial : entry.dials ){
        if( dial.collectionIndex == size_t(-1) ){ continue; }
        if( dial.interfaceIndex == size_t(-1)  ){ continue; }
        sampleIndexCacheList[entry.event.sampleIndex].back().dials.emplace_back(dial);
      }
    }

    LogInfo << "Cleaning up the index cache..." << std::endl;
    _indexedCache_.clear();

    LogInfo << "Performing per sample sorting..." << std::endl;
    int iSample{-1};
    for( auto& sample : sampleSet_.getSampleList() ){
      iSample++;

      auto p = GenericToolbox::getSortPermutation(
          sample.getMcContainer().getEventList(), []( const Event& a, const Event& b) {
            if( a.getIndices().dataset < a.getIndices().dataset ){ return true; }
            if( a.getIndices().entry < b.getIndices().entry ){ return true; }
            return false;
          });

      LogThrowIf(
          sampleIndexCacheList[iSample].size() != sample.getMcContainer().getEventList().size(),
          std::endl << "MISMATCH cache and event list for sample: #" << sample.getIndex() << " " << sample.getName()
              << std::endl << GET_VAR_NAME_VALUE(sampleIndexCacheList[iSample].size())
              << " <-> " << GET_VAR_NAME_VALUE(sample.getMcContainer().getEventList().size())
      );
      nCacheSlots += sampleIndexCacheList[iSample].size();

      GenericToolbox::applyPermutation( sample.getMcContainer().getEventList(), p );
      GenericToolbox::applyPermutation( sampleIndexCacheList[iSample],     p );

      // now update the event indices
      for( size_t iEvent = 0 ; iEvent < sample.getMcContainer().getEventList().size() ; iEvent++ ){
        sampleIndexCacheList[iSample][iEvent].event.eventIndex = iEvent;
      }
    }

  }

  auto countValidDials = [](std::vector<DialIndexCacheEntry>& dialIndices_){
    return std::count_if(dialIndices_.begin(), dialIndices_.end(),
      []( DialIndexCacheEntry& dialIndex_){
        if( dialIndex_.collectionIndex == size_t(-1) ){ return false; }
        if( dialIndex_.interfaceIndex == size_t(-1) ){ return false; }
        return true;
      });
  };

  LogInfo << "Filling up the " << nCacheSlots << " cache dial with references..." << std::endl;
  _cache_.reserve( nCacheSlots );

  for( auto& sampleIndexCache : sampleIndexCacheList ){
    for( auto& indexCache : sampleIndexCache ){
      
      _cache_.emplace_back();
      auto& cacheEntry = _cache_.back();

      cacheEntry.event =
          &sampleSet_.getSampleList().at(
              indexCache.event.sampleIndex
          ).getMcContainer().getEventList().at(
              indexCache.event.eventIndex
          );

      // make sure we don't need extra allocation while emplace_back
      cacheEntry.dialResponseCacheList.reserve( countValidDials(indexCache.dials) );

      // filling up the dial references
      for( auto& dialIndex : indexCache.dials ){
        if( dialIndex.collectionIndex == size_t(-1) or dialIndex.interfaceIndex == size_t(-1) ){ continue; }
        cacheEntry.dialResponseCacheList.emplace_back(
            dialCollectionList_.at(dialIndex.collectionIndex)
            .getDialInterfaceList().at(dialIndex.interfaceIndex)
        );
      }
    }
  }
}
void EventDialCache::allocateCacheEntries( size_t nEvent_, size_t nDialsMaxPerEvent_) {
    _indexedCache_.resize(
        _indexedCache_.size() + nEvent_,
        {{std::size_t(-1),std::size_t(-1)},
         std::vector<DialIndexCacheEntry>(nDialsMaxPerEvent_,
                                          {std::size_t(-1),std::size_t(-1)})} );
}
void EventDialCache::shrinkIndexedCache(){
  _indexedCache_.resize(_fillIndex_+1);
  _indexedCache_.shrink_to_fit();
}

EventDialCache::IndexedCacheEntry* EventDialCache::fetchNextCacheEntry(){
  // Warning warning Will Robinson!
  // This only works IFF the indexed cache is not resized.
  LogThrowIf(_fillIndex_ >= _indexedCache_.size());
  return &_indexedCache_[_fillIndex_++];
}


void EventDialCache::reweightEntry( EventDialCache::CacheEntry& entry_){
  // storing the reweight factor in a temporary buffer
  // this allows to perform capping of the value
  double tempReweight{1};

  // calculate the dial responses
  for( auto& dialResponseCache : entry_.dialResponseCacheList ){
    tempReweight *= dialResponseCache.getResponse();
  }

  // applying event weight cap if defined
  _globalEventReweightCap_.process( tempReweight );

  entry_.event->getWeights().resetCurrentWeight(); // reset to the base weight
  entry_.event->getWeights().current *= tempReweight; // apply the reweight factor
}
