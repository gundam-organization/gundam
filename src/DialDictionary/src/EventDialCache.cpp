//
// Created by Adrien Blanchet on 01/12/2022.
//

#include "EventDialCache.h"

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[EventDialCache]");
});


std::vector<std::pair<PhysicsEvent *, std::vector<DialInterface *>>> &EventDialCache::getCache() {
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

  _cache_.reserve(
      std::count_if(
          _indexedCache_.begin(), _indexedCache_.end(),
          isCacheEntryValid)
      );

  for(auto & entry : _indexedCache_){

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

    for( auto& dialIndex : entry.second ){
      if( dialIndex.first == size_t(-1) or dialIndex.second == size_t(-1) ){ continue; }
      _cache_.back().second.emplace_back(
          &dialCollectionList_.at(dialIndex.first).getDialInterfaceList().at(dialIndex.second)
      );
    }
  }
}
void EventDialCache::allocateCacheEntries(size_t nEvent_, size_t nDialsMaxPerEvent_) {
  _indexedCache_.resize( _indexedCache_.size() + nEvent_, {{-1,-1}, std::vector<std::pair<size_t, size_t>>(nDialsMaxPerEvent_, {-1,-1})} );
}
std::pair<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>>* EventDialCache::fetchNextCacheEntry(){
#if __cplusplus >= 201703L
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

void EventDialCache::reweightEntry(std::pair<PhysicsEvent*, std::vector<DialInterface*>>& entry_){
  entry_.first->resetEventWeight();
  std::for_each(entry_.second.begin(), entry_.second.end(), [&](DialInterface* dial_){
    entry_.first->getEventWeightRef() *= dial_->evalResponse();
  });
}

