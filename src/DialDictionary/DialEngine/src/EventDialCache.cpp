//
// Created by Adrien Blanchet on 01/12/2022.
//

#include "EventDialCache.h"

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[EventDialCache]");
});


std::vector<EventDialCache::CacheElem_t> &EventDialCache::getCache() {
  return _cache_;
}
const std::vector<EventDialCache::CacheElem_t> &EventDialCache::getCache() const {
  return _cache_;
}

void EventDialCache::buildReferenceCache(FitSampleSet& sampleSet_, std::vector<DialCollection>& dialCollectionList_){
  LogInfo << "Building event dial cache..." << std::endl;

  LogInfo << "Sorting events in sync with indexed cache..." << std::endl;

  {
    LogScopeIndent;
    LogInfo << "Breaking down indexed cache per sample..." << std::endl;
    std::vector<std::vector<IndexedEntry_t>> sampleIndexCacheList{sampleSet_.getFitSampleList().size()};
    for( auto& entry : _indexedCache_ ){
      if( entry.event.sampleIndex == size_t(-1) ){ continue; }
      if( entry.event.eventIndex == size_t(-1) ){ continue; }
      sampleIndexCacheList[entry.event.sampleIndex].emplace_back( entry );
    }
    _indexedCache_.clear();

    LogInfo << "Performing per sample sorting..." << std::endl;
    int iSample{-1};
    for( auto& sample : sampleSet_.getFitSampleList() ){
      iSample++;

      auto p = GenericToolbox::getSortPermutation(
          sample.getMcContainer().eventList, [](const PhysicsEvent& a, const PhysicsEvent& b) {
            if( a.getDataSetIndex() < b.getDataSetIndex() ){ return true; }
            if( a.getEntryIndex() < b.getEntryIndex() ){ return true; }
            return false;
          });


      LogDebug << GET_VAR_NAME_VALUE(sample.getName()) << std::endl;
      LogDebug << GET_VAR_NAME_VALUE(sampleIndexCacheList[iSample].size()) << std::endl;
      LogDebug << GET_VAR_NAME_VALUE(sample.getMcContainer().eventList.size()) << std::endl;

      GenericToolbox::applyPermutation( sample.getMcContainer().eventList, p );

      GenericToolbox::applyPermutation( sampleIndexCacheList[iSample],   p );

      // now update the event indices
      for( size_t iEvent = 0 ; iEvent < sample.getMcContainer().eventList.size() ; iEvent++ ){
        sampleIndexCacheList[iSample][iEvent].event.eventIndex = iEvent;
      }
    }

    LogInfo << "Propagating per sample indexed cache to the full indexed cache..." << std::endl;
    for( auto& sampleIndexCache : sampleIndexCacheList ){
      for( auto& entry : sampleIndexCache ){
        _indexedCache_.emplace_back( entry );
      }
    }
  }

  auto countValidDials = [](std::vector<DialIndexEntry_t>& dialIndices_){
    return std::count_if(dialIndices_.begin(), dialIndices_.end(),
      [](DialIndexEntry_t& dialIndex_){
        if( dialIndex_.collectionIndex == size_t(-1) ){ return false; }
        if( dialIndex_.interfaceIndex == size_t(-1) ){ return false; }
        return true;
      });
  };

//  auto isCacheEntryValid = [&](IndexedEntry_t& entry_){
//    if( entry_.event.sampleIndex == size_t(-1) ){ return false; }
//    if( entry_.event.eventIndex == size_t(-1) ){ return false; }
//    if( entry_.dials.empty() ){ return false; }
//    return countValidDials(entry_.dials) != 0;
//  };

  // reserve memory before emplace_back
//  long nCacheEntries = std::count_if(
//      _indexedCache_.begin(), _indexedCache_.end(),
//      isCacheEntryValid
//  );
  _cache_.reserve( _indexedCache_.size() );

  for( auto& entry : _indexedCache_ ){

//    if( not isCacheEntryValid(entry) ){
//      if( _warnForDialLessEvent_ ){
//        LogAlert << "Sample \"" << sampleSet_.getFitSampleList().at(entry.event.sampleIndex).getName() << "\"";
//        LogAlert << "/Event #" << entry.event.eventIndex << " has no dial!" << std::endl;
//      }
//      continue;
//    }

    _cache_.emplace_back();
    _cache_.back().event =
        &sampleSet_.getFitSampleList().at(
            entry.event.sampleIndex
        ).getMcContainer().eventList.at(
            entry.event.eventIndex
        );

    _cache_.back().dials.reserve( countValidDials(entry.dials) );
    for( auto& dialIndex : entry.dials ){
      if( dialIndex.collectionIndex == size_t(-1) or dialIndex.interfaceIndex == size_t(-1) ){ continue; }
#ifndef USE_BREAKDOWN_CACHE
      _cache_.back().dials.emplace_back(
          &dialCollectionList_.at(dialIndex.collectionIndex)
          .getDialInterfaceList().at(dialIndex.interfaceIndex)
      );
#else
      _cache_.back().dials.emplace_back(
          &dialCollectionList_.at(dialIndex.collectionIndex)
          .getDialInterfaceList().at(dialIndex.interfaceIndex),
          std::nan("unset")
      );
#endif
    }
  }

}

void EventDialCache::allocateCacheEntries(size_t nEvent_, size_t nDialsMaxPerEvent_) {
    _indexedCache_.resize(
        _indexedCache_.size() + nEvent_,
        {{std::size_t(-1),std::size_t(-1)},
         std::vector<DialIndexEntry_t>(nDialsMaxPerEvent_,
                                       {std::size_t(-1),std::size_t(-1)})} );
}

EventDialCache::IndexedEntry_t* EventDialCache::fetchNextCacheEntry(){
#if HAS_CPP_17
  std::scoped_lock<std::mutex> g(_mutex_);
#else
  std::lock_guard<std::mutex> g(_mutex_);
#endif
  // This is VERY not thread safe since another thread could emplace a new
  // value on the back of the indexed cache and force the vector to be copied.
  // I don't see where the space for the cache is reserved (there is a resize,
  // but that doesn't solve this problem.
  if( _fillIndex_ >= _indexedCache_.size() ){
    LogThrow("out of range: " << _fillIndex_);
    _indexedCache_.emplace_back();
  }
  // Warning warning Will Robinson!  This only works IFF the indexed cache is
  // not resized (violated by the previouls stanza).
  return &_indexedCache_[_fillIndex_++];
}

void EventDialCache::shrinkIndexedCache(){
  _indexedCache_.resize(_fillIndex_+1);
  _indexedCache_.shrink_to_fit();
}

#ifndef USE_BREAKDOWN_CACHE
void EventDialCache::reweightEntry(EventDialCache::CacheElem_t& entry_){
  entry_.event->resetEventWeight();
  std::for_each(entry_.dials.begin(), entry_.dials.end(), [&](DialInterface* dial_){
    entry_.event->getEventWeightRef() *= dial_->evalResponse();
  });
}
#else
void EventDialCache::reweightEntry(EventDialCache::CacheElem_t& entry_){
  entry_.event->resetEventWeight();
  std::for_each(entry_.dials.begin(), entry_.dials.end(), [&](DialsElem_t& dial_){
    if( dial_.interface->getInputBufferRef()->isMasked() ){ return ; }
    if(std::isnan(dial_.response) or dial_.interface->getInputBufferRef()->isDialUpdateRequested() ){
      dial_.response = dial_.interface->evalResponse();
    }

    if( std::isnan( dial_.response ) ){
      LogError << "Invalid dial response:" << std::endl;
      LogError << dial_.interface->getSummary(false ) << std::endl;
      LogError << GET_VAR_NAME_VALUE( dial_.interface->evalResponse() ) << std::endl;
      LogThrow("Exception thrown because of invalid spline response.");
    }

    LogThrowIf(std::isnan(dial_.response), "Invalid dial response for " << dial_.interface->getSummary());
    entry_.event->getEventWeightRef() *= dial_.response;
  });
}
#endif
