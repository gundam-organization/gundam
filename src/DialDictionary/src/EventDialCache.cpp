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

void EventDialCache::buildReferenceCache(FitSampleSet& sampleSet_, std::vector<DialCollection>& dialCollectionList_){
  LogInfo << "Building event dial cache..." << std::endl;

  auto countValidDials = [](std::vector<std::pair<size_t, size_t>>& dialIndices_){
    return std::count_if(dialIndices_.begin(), dialIndices_.end(),
      [](std::pair<size_t, size_t>& dialIndex_){
        if( dialIndex_.first == size_t(-1) or dialIndex_.second == size_t(-1) ){ return false; }
        return true;
      });
  };

  auto isCacheEntryValid = [&](std::pair<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>>& entry_){
    if( entry_.first.first == size_t(-1) or entry_.first.second == size_t(-1) or entry_.second.empty() ){ return false; }
    return countValidDials(entry_.second) != 0;
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
        LogAlert << "Sample \"" << sampleSet_.getFitSampleList().at(entry.first.first).getName() << "\"";
        LogAlert << "/Event #" << entry.first.second << " has no dial!" << std::endl;
      }
      continue;
    }

    _cache_.emplace_back();
    _cache_.back().event =
        &sampleSet_.getFitSampleList().at(
            entry.first.first
        ).getMcContainer().eventList.at(
            entry.first.second
        );

    _cache_.back().dials.reserve( countValidDials(entry.second) );

//    LogTrace << "Sample #" << entry.first.first << "(" << sampleSet_.getFitSampleList().at(entry.first.first).getName() << ") / MC_Event#" << entry.first.second << std::endl;
//    LogTrace << "Nb of dials: " << entry.second.size() << std::endl;
//    LogTrace << "dials :" << GenericToolbox::iterableToString(entry.second, [&](auto& dialIndex){
//      std::stringstream ss;
//      if( dialIndex.first == size_t(-1) or dialIndex.second == size_t(-1) ){ return std::string("invalid"); }
//      ss << dialCollectionList_.at(dialIndex.first).getTitle() << "/#" << dialIndex.second;
//      ss << dialIndex.first << "/#" << dialIndex.second;
//      return ss.str();
//    }) << std::endl;
    for( auto& dialIndex : entry.second ){
      if( dialIndex.first == size_t(-1) or dialIndex.second == size_t(-1) ){ continue; }
#ifndef USE_BREAKDOWN_CACHE
      _cache_.back().dials.emplace_back(
          &dialCollectionList_.at(dialIndex.first).getDialInterfaceList().at(dialIndex.second)
      );
#else
      _cache_.back().dials.emplace_back(
          &dialCollectionList_.at(dialIndex.first).getDialInterfaceList().at(dialIndex.second), std::nan("unset")
      );
#endif
    }
  }
}
void EventDialCache::allocateCacheEntries(size_t nEvent_, size_t nDialsMaxPerEvent_) {
  _indexedCache_.resize( _indexedCache_.size() + nEvent_, {{-1,-1}, std::vector<std::pair<size_t, size_t>>(nDialsMaxPerEvent_, {-1,-1})} );
}
std::pair<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>>* EventDialCache::fetchNextCacheEntry(){
#if HAS_CPP_17
  std::scoped_lock<std::mutex> g(_mutex_);
#else
  std::lock_guard<std::mutex> g(_mutex_);
#endif
  if( _fillIndex_ >= _indexedCache_.size() ){
    LogThrow("out of range: " << _fillIndex_);
    _indexedCache_.emplace_back();
  }
  return &_indexedCache_[_fillIndex_++];
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
    if( dial_.dial->getInputBufferRef()->isMasked() ){ return ; }
    if( std::isnan(dial_.result) or dial_.dial->getInputBufferRef()->isDialUpdateRequested() ){
      dial_.result = dial_.dial->evalResponse();
    }
    LogThrowIf(std::isnan(dial_.result), "Invalid dial response for " << dial_.dial->getSummary());
    entry_.event->getEventWeightRef() *= dial_.result;
  });
}
#endif
