//
// Created by Adrien Blanchet on 01/12/2022.
//

#include "EventDialCache.h"

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[EventDialCache]");
});


void EventDialCache::buildReferenceCache(FitSampleSet& sampleSet_, std::vector<DialCollection>& dialCollectionList_){


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

    LogTrace << entry.first.first << " / " << entry.first.second << std::endl;

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
      LogTrace << " -> " << dialIndex.first << " / " << dialIndex.second << "  ";
      LogTrace << dialCollectionList_.at(dialIndex.first).getSummary();
      LogTrace << "  " << dialCollectionList_.at(dialIndex.first).getDialBaseList().size() << std::endl;

      _cache_.back().second.emplace_back(
          &dialCollectionList_.at(dialIndex.first).getDialInterfaceList().at(dialIndex.second)
      );
    }

    LogThrowIf(_cache_.back().second.empty(), "empty");
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
void EventDialCache::propagate(int iThread_, int nThreads_){

  auto start = _cache_.begin();
  auto end = _cache_.end();

  if( nThreads_ != 1 ){
    Long64_t nEventPerThread = Long64_t(_cache_.size())/nThreads_;
    start = _cache_.begin() + Long64_t(iThread_)*nEventPerThread;
    if( iThread_+1 != GlobalVariables::getNbThreads() ){
      end = _cache_.begin() + (Long64_t(iThread_) + 1) * nEventPerThread;
    }
  }

  double weightBuffer{std::nan("unset")};
  std::for_each(start, end, [&](std::pair<PhysicsEvent*, std::vector<DialInterface*>>& entry_){
    weightBuffer = entry_.first->getTreeWeight();
    std::for_each(entry_.second.begin(), entry_.second.end(), [&](DialInterface* dial_){
      weightBuffer *= dial_->evalResponse();
    });
    entry_.first->setEventWeight(weightBuffer);
  });

}

void EventDialCache::sortCache(){
  std::sort(_cache_.begin(), _cache_.end(), [](
      const std::pair<PhysicsEvent*, std::vector<DialInterface*>>& a_,
      const std::pair<PhysicsEvent*, std::vector<DialInterface*>>& b_){
    // a goes first?
    return a_.first < b_.first;
  });

  std::for_each(_cache_.begin(), _cache_.end(),[]( std::pair<PhysicsEvent*, std::vector<DialInterface*>>& entry_ ){
    std::sort(entry_.second.begin(), entry_.second.end(), []( const DialInterface* a_, const DialInterface* b_ ){
      // a goes first?
      return a_ < b_;
    });
  });
}
