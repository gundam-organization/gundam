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

  auto countValidDials = [](std::vector<DialIndexEntry_t>& dialIndices_){
    return std::count_if(dialIndices_.begin(), dialIndices_.end(),
      [](DialIndexEntry_t& dialIndex_){
        if( dialIndex_.collectionIndex == size_t(-1) ){ return false; }
        if( dialIndex_.interfaceIndex == size_t(-1) ){ return false; }
        return true;
      });
  };

  auto isCacheEntryValid = [&](IndexedEntry_t& entry_){
    if( entry_.event.sampleIndex == size_t(-1) ){ return false; }
    if( entry_.event.eventIndex == size_t(-1) ){ return false; }
    if( entry_.dials.empty() ){ return false; }
    return countValidDials(entry_.dials) != 0;
  };

  // reserve memory before emplace_back
  long nCacheEntries = std::count_if(
      _indexedCache_.begin(), _indexedCache_.end(),
      isCacheEntryValid
  );
  _cache_.reserve( nCacheEntries );

  for( auto& entry : _indexedCache_ ){

    if( not isCacheEntryValid(entry) ){
      if( _warnForDialLessEvent_ ){
        LogAlert << "Sample \"" << sampleSet_.getFitSampleList().at(entry.event.sampleIndex).getName() << "\"";
        LogAlert << "/Event #" << entry.event.eventIndex << " has no dial!" << std::endl;
      }
      continue;
    }

    _cache_.emplace_back();
    _cache_.back().event =
        &sampleSet_.getFitSampleList().at(
            entry.event.sampleIndex
        ).getMcContainer().eventList.at(
            entry.event.eventIndex
        );

    _cache_.back().dials.reserve( countValidDials(entry.dials) );

//    LogTrace << "Sample #" << entry.first.first << "(" << sampleSet_.getFitSampleList().at(entry.first.first).getName() << ") / MC_Event#" << entry.first.second << std::endl;
//    LogTrace << "Nb of dials: " << entry.second.size() << std::endl;
//    LogTrace << "dials :" << GenericToolbox::iterableToString(entry.second, [&](auto& dialIndex){
//      std::stringstream ss;
//      if( dialIndex.first == size_t(-1) or dialIndex.second == size_t(-1) ){ return std::string("invalid"); }
//      ss << dialCollectionList_.at(dialIndex.first).getTitle() << "/#" << dialIndex.second;
//      ss << dialIndex.first << "/#" << dialIndex.second;
//      return ss.str();
//    }) << std::endl;
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

  LogInfo << "Re-sorting event cache entries..." << std::endl;
  auto p = GenericToolbox::getSortPermutation(_cache_, [](const CacheElem_t& a, const CacheElem_t& b){
    if( a.event->getDataSetIndex() < b.event->getDataSetIndex() ) { return true; }
    if( a.event->getEntryIndex()   < b.event->getEntryIndex()   ) { return true; }
    return false;
  });

  GenericToolbox::applyPermutation(_cache_, p);

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
