//
// Created by Adrien Blanchet on 01/12/2022.
//

#include "EventDialCache.h"

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[EventDialCache]");
});

double EventDialCache::globalEventReweightCap{std::nan("unset")};

void EventDialCache::buildReferenceCache(SampleSet& sampleSet_, std::vector<DialCollection>& dialCollectionList_){
  LogInfo << "Building event dial cache..." << std::endl;

  LogInfo << "Sorting events in sync with indexed cache..." << std::endl;

  size_t nCacheSlots{0};
  std::vector<std::vector<IndexedEntry_t>> sampleIndexCacheList{sampleSet_.getFitSampleList().size()};

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

      LogThrowIf(
          sampleIndexCacheList[iSample].size() != sample.getMcContainer().eventList.size(),
          std::endl << "MISMATCH cache and event list for sample: #" << sample.getIndex() << " " << sample.getName()
              << std::endl << GET_VAR_NAME_VALUE(sampleIndexCacheList[iSample].size())
              << " <-> " << GET_VAR_NAME_VALUE(sample.getMcContainer().eventList.size())
      );
      nCacheSlots += sampleIndexCacheList[iSample].size();

      GenericToolbox::applyPermutation( sample.getMcContainer().eventList, p );
      GenericToolbox::applyPermutation( sampleIndexCacheList[iSample],     p );

      // now update the event indices
      for( size_t iEvent = 0 ; iEvent < sample.getMcContainer().eventList.size() ; iEvent++ ){
        sampleIndexCacheList[iSample][iEvent].event.eventIndex = iEvent;
      }
    }

//    LogInfo << "Propagating per sample indexed cache to the full indexed cache..." << std::endl;
//    for( auto& sampleIndexCache : sampleIndexCacheList ){
//      _indexedCache_.reserve( _indexedCache_.size() + sampleIndexCache.size() );
//      for( auto& entry : sampleIndexCache ){
//        _indexedCache_.emplace_back( entry );
//      }
//    }
  }

  auto countValidDials = [](std::vector<DialIndexEntry_t>& dialIndices_){
    return std::count_if(dialIndices_.begin(), dialIndices_.end(),
      [](DialIndexEntry_t& dialIndex_){
        if( dialIndex_.collectionIndex == size_t(-1) ){ return false; }
        if( dialIndex_.interfaceIndex == size_t(-1) ){ return false; }
        return true;
      });
  };

  LogInfo << "Filling up the cache dial with references..." << std::endl;
  _cache_.reserve( nCacheSlots );

  for( auto& sampleIndexCache : sampleIndexCacheList ){
    for( auto& indexCache : sampleIndexCache ){
      _cache_.emplace_back();

      auto& cacheEntry{_cache_.back()};

      cacheEntry.event =
          &sampleSet_.getFitSampleList().at(
              indexCache.event.sampleIndex
          ).getMcContainer().eventList.at(
              indexCache.event.eventIndex
          );

      cacheEntry.dials.reserve( countValidDials(indexCache.dials) );
      for( auto& dialIndex : indexCache.dials ){
        if( dialIndex.collectionIndex == size_t(-1) or dialIndex.interfaceIndex == size_t(-1) ){ continue; }
        cacheEntry.dials.emplace_back(
            &dialCollectionList_.at(dialIndex.collectionIndex)
                .getDialInterfaceList().at(dialIndex.interfaceIndex),
            std::nan("unset")
        );
      }
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
void EventDialCache::shrinkIndexedCache(){
  _indexedCache_.resize(_fillIndex_+1);
  _indexedCache_.shrink_to_fit();
}

EventDialCache::IndexedEntry_t* EventDialCache::fetchNextCacheEntry(){
  // for debug, uncomment:
//  LogThrowIf(
//      _fillIndex_ >= _indexedCache_.size(),
//      "out of range: " << GET_VAR_NAME_VALUE(_fillIndex_)
//      << " while: " << GET_VAR_NAME_VALUE(_indexedCache_.size())
//  );

  // Warning warning Will Robinson!
  // This only works IFF the indexed cache is not resized.
  return &_indexedCache_[_fillIndex_++];
}


void EventDialCache::reweightEntry(EventDialCache::CacheElem_t& entry_){
  double tempReweight{1};

  // calculate the dial responses
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
    tempReweight *= dial_.response;
  });

  // Applying event weight cap
  if( not std::isnan(EventDialCache::globalEventReweightCap) ){
    if( tempReweight > EventDialCache::globalEventReweightCap ){
      tempReweight = EventDialCache::globalEventReweightCap;
    }
  }

  entry_.event->resetEventWeight(); // reset to the base weight
  entry_.event->getEventWeightRef() *= tempReweight; // apply the reweight factor
}
