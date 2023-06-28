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
      sampleIndexCacheList[entry.event.sampleIndex].back().dials.clear();
      for( auto& dial : entry.dials ){
        if( dial.collectionIndex == size_t(-1) ){ continue; }
        if( dial.interfaceIndex == size_t(-1)  ){ continue; }
        sampleIndexCacheList[entry.event.sampleIndex].back().dials.emplace_back(dial);
      }

    }
    _indexedCache_.clear();

    std::stringstream ssUnsorted;
    for( auto& sampleIndexCache : sampleIndexCacheList ){
      for( auto& entry : sampleIndexCache ){
        auto& ev = sampleSet_.getFitSampleList().at(
            entry.event.sampleIndex
        ).getMcContainer().eventList.at(
            entry.event.eventIndex
        );

        ssUnsorted << entry.event.sampleIndex << "/" << entry.event.eventIndex << " -> ";
        ssUnsorted << ev.getEntryIndex() << " =dials=> ";

        ssUnsorted << GenericToolbox::iterableToString(sampleIndexCacheList[entry.event.sampleIndex][entry.event.eventIndex].dials, [](const DialIndexEntry_t& e){
          return "{" + std::to_string(e.collectionIndex) + ", " +std::to_string(e.interfaceIndex) + "}";
        }, false) << std::endl;
      }
    }
    GenericToolbox::dumpStringInFile("./indexedCacheUnsorted.txt", ssUnsorted.str());


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


      LogThrowIf(
          sampleIndexCacheList[iSample].size() != sample.getMcContainer().eventList.size(),
          "MISMATCH cache and event list"
      );

      GenericToolbox::applyPermutation( sample.getMcContainer().eventList, p );
      GenericToolbox::applyPermutation( sampleIndexCacheList[iSample],     p );

      // now update the event indices
      for( size_t iEvent = 0 ; iEvent < sample.getMcContainer().eventList.size() ; iEvent++ ){
        sampleIndexCacheList[iSample][iEvent].event.eventIndex = iEvent;
      }
    }

    LogInfo << "Propagating per sample indexed cache to the full indexed cache..." << std::endl;
    std::stringstream ss;
    for( auto& sampleIndexCache : sampleIndexCacheList ){
      _indexedCache_.reserve( _indexedCache_.size() + sampleIndexCache.size() );
      for( auto& entry : sampleIndexCache ){
        _indexedCache_.emplace_back( entry );

        auto& ev = sampleSet_.getFitSampleList().at(
            entry.event.sampleIndex
        ).getMcContainer().eventList.at(
            entry.event.eventIndex
        );

        ss << entry.event.sampleIndex << "/" << entry.event.eventIndex << " -> ";
        ss << ev.getEntryIndex() << " =dials=> ";
        ss << GenericToolbox::iterableToString(entry.dials, [](const DialIndexEntry_t& e){
          return "{" + std::to_string(e.collectionIndex) + ", " +std::to_string(e.interfaceIndex) + "}";
        }, false) << std::endl;
      }
    }
    GenericToolbox::dumpStringInFile("./indexedCache.txt", ss.str());
  }

  auto countValidDials = [](std::vector<DialIndexEntry_t>& dialIndices_){
    return std::count_if(dialIndices_.begin(), dialIndices_.end(),
      [](DialIndexEntry_t& dialIndex_){
        if( dialIndex_.collectionIndex == size_t(-1) ){ return false; }
        if( dialIndex_.interfaceIndex == size_t(-1) ){ return false; }
        return true;
      });
  };

  std::stringstream ssRef;
  _cache_.reserve( _indexedCache_.size() );
  for( auto& entry : _indexedCache_ ){

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

    ssRef << entry.event.sampleIndex << "/" << entry.event.eventIndex << " -> ";
    ssRef << _cache_.back().event->getEntryIndex() << " =dials=> ";
    ssRef << GenericToolbox::iterableToString(_cache_.back().dials, [](const DialsElem_t& e){
      return e.interface->getSummary();
    }, false) << std::endl;
  }
  GenericToolbox::dumpStringInFile("./refCache.txt", ssRef.str());

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
  LogThrowIf(
      _fillIndex_ >= _indexedCache_.size(),
      "out of range: " << GET_VAR_NAME_VALUE(_fillIndex_)
      << " while: " << GET_VAR_NAME_VALUE(_indexedCache_.size())
  );

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
