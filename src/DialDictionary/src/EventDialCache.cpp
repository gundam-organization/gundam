//
// Created by Adrien Blanchet on 01/12/2022.
//

#include "EventDialCache.h"

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[EventDialCache]");
});


#ifndef USE_BREAKDOWN_CACHE
std::vector<std::pair<PhysicsEvent *, std::vector<DialInterface *>>> &EventDialCache::getCache() {
  return _cache_;
}
#else
std::vector<std::pair< PhysicsEvent*, std::vector<std::pair<DialInterface*, double> > >> &EventDialCache::getCache() {
  return _cache_;
}
#endif

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
    _cache_.back().first =
        &sampleSet_.getFitSampleList().at(
            entry.first.first
        ).getMcContainer().eventList.at(
            entry.first.second
        );

    _cache_.back().second.reserve( countValidDials(entry.second) );

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
      _cache_.back().second.emplace_back(
          &dialCollectionList_.at(dialIndex.first).getDialInterfaceList().at(dialIndex.second)
      );
#else
      _cache_.back().second.emplace_back(
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
    LogThrow("out of range" << _fillIndex_);
    _indexedCache_.emplace_back();
  }
  return &_indexedCache_[_fillIndex_++];
}

#ifndef USE_BREAKDOWN_CACHE
void EventDialCache::reweightEntry(std::pair<PhysicsEvent*, std::vector<DialInterface*>>& entry_){
  entry_.first->resetEventWeight();
  std::for_each(entry_.second.begin(), entry_.second.end(), [&](DialInterface* dial_){
    entry_.first->getEventWeightRef() *= dial_->evalResponse();
  });
}
#else
void EventDialCache::reweightEntry(std::pair<PhysicsEvent*, std::vector<std::pair<DialInterface*, double>>>& entry_){
  entry_.first->resetEventWeight();
  std::for_each(entry_.second.begin(), entry_.second.end(), [&](std::pair<DialInterface*, double>& dial_){
    if( std::isnan(dial_.second) or dial_.first->getInputBufferRef()->isDialUpdateRequested() ){
      dial_.second = dial_.first->evalResponse();
    }
    LogThrowIf(std::isnan(dial_.second), "Invalid dial response for " << dial_.first->getSummary());
    entry_.first->getEventWeightRef() *= dial_.second;
  });
}
#endif
