//
// Created by Adrien Blanchet on 01/12/2022.
//

#include "EventDialCache.h"

#include "Logger.h"
#include "Parameter.h"

#include <algorithm>

#include <algorithm>


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
          sample.getEventList(), []( const Event& a, const Event& b) {
            if( a.getIndices().dataset < a.getIndices().dataset ){ return true; }
            if( a.getIndices().entry < b.getIndices().entry ){ return true; }
            return false;
          });

      LogThrowIf(
          sampleIndexCacheList[iSample].size() != sample.getEventList().size(),
          std::endl << "MISMATCH cache and event list for sample: #" << sample.getIndex() << " " << sample.getName()
              << std::endl << GET_VAR_NAME_VALUE(sampleIndexCacheList[iSample].size())
              << " <-> " << GET_VAR_NAME_VALUE(sample.getEventList().size())
      );
      nCacheSlots += sampleIndexCacheList[iSample].size();

      GenericToolbox::applyPermutation( sample.getEventList(), p );
      GenericToolbox::applyPermutation( sampleIndexCacheList[iSample],     p );

      // now update the event indices
      for( size_t iEvent = 0 ; iEvent < sample.getEventList().size() ; iEvent++ ){
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
          ).getEventList().at(
              indexCache.event.eventIndex
          );

      // make sure we don't need extra allocation while emplace_back
      cacheEntry.dialResponseCacheList.reserve( countValidDials(indexCache.dials) );

      // filling up the dial references
      for( auto& dialIndex : indexCache.dials ){
        if( dialIndex.collectionIndex == size_t(-1) or dialIndex.interfaceIndex == size_t(-1) ){ continue; }

        if( dialIndex.interfaceIndex >= dialCollectionList_.at(dialIndex.collectionIndex).getDialInterfaceList().size() ){
          LogError << "BAD: dialIndex.interfaceIndex >= dialCollectionList_.at(dialIndex.collectionIndex).getDialInterfaceList().size()" << std::endl;
          LogError << GET_VAR_NAME_VALUE(dialIndex.interfaceIndex) << " (+1 for the size)" << std::endl;
          LogError << "Selected collection: " << dialCollectionList_.at(dialIndex.collectionIndex).getSummary() << std::endl;
          LogError << "Nb of defined interfaces: " << dialCollectionList_.at(dialIndex.collectionIndex).getDialInterfaceList().size() << std::endl;

          LogError << "Listing available collections:" << std::endl;
          for( auto& dialCol : dialCollectionList_ ){
            LogDebug << dialCol.getSummary() << std::endl;
          }

          LogThrow("DEV ERROR: Please report this issue to github!! This should not happen");
        }

        cacheEntry.dialResponseCacheList.emplace_back(
            dialCollectionList_.at(dialIndex.collectionIndex)
            .getDialInterfaceList().at(dialIndex.interfaceIndex)
        );
        const auto& collection = dialCollectionList_.at(dialIndex.collectionIndex);
        if( collection.getApplyToVariable().empty() ){ continue; }
        _hasVariableResponses_ = true;
        cacheEntry.dialResponseCacheList.back().appliesToVariable = true;
        auto& variables = cacheEntry.event->getVariables();
        int variableIndex = variables.findVarIndex(collection.getApplyToVariable());
        auto& responses = cacheEntry.variableResponseCacheList;
        auto target = std::find_if(responses.begin(), responses.end(), [&](const VariableResponseCache& response_){
          return response_.variableIndex == variableIndex;
        });
        if( target == responses.end() ){
          responses.emplace_back();
          target = responses.end() - 1;
          target->variableIndex = variableIndex;
          target->referenceValue = variables.getVarList()[variableIndex].getVarAsDouble();
        }
        auto responseIndex = cacheEntry.dialResponseCacheList.size() - 1;
        auto mode = collection.getVariableResponseMode();
        if( mode == DialCollection::VariableResponseMode::Multiply ){ target->multiplyDials.push_back(responseIndex); }
        else if( mode == DialCollection::VariableResponseMode::Add ){ target->addDials.push_back(responseIndex); }
        else{
          LogThrowIf(target->overrideDial != std::size_t(-1),
                     "Multiple Override dials for variable \"" << collection.getApplyToVariable()
                     << "\" on event " << *cacheEntry.event);
          target->overrideDial = responseIndex;
        }
      }
    }
  }

  LogInfo << "Reference cache has been setup." << std::endl;
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
void EventDialCache::fillCacheEntries(const SampleSet& sampleSet_){
  _fillIndex_ = 0;
  allocateCacheEntries( sampleSet_.getNbOfEvents(), 0 );
  for( size_t iSample = 0 ; iSample < sampleSet_.getSampleList().size() ; iSample++ ){
    for( size_t iEvent = 0 ; iEvent < sampleSet_.getSampleList()[iSample].getEventList().size() ; iEvent++ ){
      auto* entry = fetchNextCacheEntry();
      entry->event.sampleIndex = iSample;
      entry->event.eventIndex = iEvent;
    }
  }
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
    if( dialResponseCache.appliesToVariable ){ continue; }
    tempReweight *= dialResponseCache.getResponse();
  }

  // applying event weight cap if defined
  _globalEventReweightCap_.process( tempReweight );

  entry_.event->getWeights().resetCurrentWeight(); // reset to the base weight
  entry_.event->getWeights().current *= tempReweight; // apply the reweight factor
}

void EventDialCache::applyVariableResponses(CacheEntry& entry_){
  for( auto& target : entry_.variableResponseCacheList ){
    double value = target.overrideDial == std::size_t(-1) ? target.referenceValue
        : entry_.dialResponseCacheList[target.overrideDial].getResponse();
    for( auto index : target.multiplyDials ){ value *= entry_.dialResponseCacheList[index].getResponse(); }
    for( auto index : target.addDials ){ value += entry_.dialResponseCacheList[index].getResponse(); }
    // Dial responses are continuous values, including when the source branch was integral.
    entry_.event->getVariables().getVarList()[target.variableIndex].set(value);
  }
}

void EventDialCache::evalEventWeightGradient(EventDialCache::CacheEntry& entry_,
                                             const std::vector<Parameter*>& parameterList_,
                                             bool resetGradients_,
                                             double gradientScale_){
  if( resetGradients_ ){
    for( auto* parameter : parameterList_ ){
      if( parameter == nullptr ){ continue; }
      parameter->resetGradient();
    }
  }

  if( entry_.event == nullptr or entry_.dialResponseCacheList.empty() ){ return; }

  auto isRequestedParameter = [&](Parameter* parameter_){
    if( parameter_ == nullptr ){ return false; }
    return std::find(parameterList_.begin(), parameterList_.end(), parameter_) != parameterList_.end();
  };

  std::vector<double> responseList(entry_.dialResponseCacheList.size(), 1.);
  std::vector<double> prefix(responseList.size() + 1, 1.);
  for( size_t iDial = 0; iDial < responseList.size(); ++iDial ){
    responseList[iDial] = entry_.dialResponseCacheList[iDial].getResponse();
    prefix[iDial + 1] = prefix[iDial] * responseList[iDial];
  }
  const double responseProduct = prefix.back();

  if( _globalEventReweightCap_.isEnabled and responseProduct > _globalEventReweightCap_.maxReweight ){
    return;
  }

  auto addGradient = [&](Parameter* parameter_, double gradient_){
    if( parameter_ == nullptr or gradient_ == 0 ){ return; }
    if( not isRequestedParameter(parameter_) ){ return; }
    parameter_->addGradient(gradient_);
  };

  const double baseWeight{entry_.event->getWeights().base};
  double suffix = 1.;
  for( size_t iDial = responseList.size(); iDial-- > 0; ){
    // Prefix/suffix products handle zero responses without division and avoid O(D^2) work.
    const double productExcludingDial = prefix[iDial] * suffix;
    suffix *= responseList[iDial];
    auto& dialResponseCache = entry_.dialResponseCacheList[iDial];
    auto* dialInterface = dialResponseCache.dialInterface;
    if( dialInterface == nullptr or dialInterface->getInputBufferRef() == nullptr ){ continue; }

    const auto* inputBuffer = dialInterface->getInputBufferRef();
    for( int iInput = 0 ; iInput < inputBuffer->getInputSize() ; iInput++ ){
      auto* parameterPtr = &inputBuffer->getParameter(iInput);
      const double dialGradient = dialInterface->evalGradient(iInput);
      const double inputGradient = inputBuffer->evalInputGradient(iInput);
      addGradient(parameterPtr, gradientScale_ * baseWeight * productExcludingDial * dialGradient * inputGradient);
    }
  }
}


std::vector<double> EventDialCache::evalEventWeightGradients(
    const std::vector<Parameter*>& parameterList_, const std::vector<double>& eventScales_,
    GenericToolbox::ParallelWorker& workers_){
  LogThrowIf(eventScales_.size() != _cache_.size(), "Event gradient scale size mismatch.");
  const int threadCount = workers_.getNbThreads();
  std::unordered_map<const Parameter*, std::size_t> parameterIndices;
  parameterIndices.reserve(parameterList_.size());
  for( std::size_t i = 0; i < parameterList_.size(); ++i ){
    if( parameterList_[i] != nullptr ){ parameterIndices.emplace(parameterList_[i], i); }
  }
  const auto& indices = parameterIndices; // Concurrent read-only lookup.
  std::vector<std::vector<double>> partials(threadCount, std::vector<double>(parameterList_.size(), 0.));
  workers_.runJob([&](int thread_){
    auto& sums = partials.at(thread_);
    // Scratch storage is reused for every event assigned to this worker.
    std::vector<double> responses, prefix;
    const auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(thread_, threadCount, _cache_.size());
    for( auto eventIndex = bounds.beginIndex; eventIndex < bounds.endIndex; ++eventIndex ){
      const double scale = eventScales_[eventIndex];
      auto& entry = _cache_[eventIndex];
      if( scale == 0. or entry.event == nullptr ){ continue; }
      const auto count = entry.dialResponseCacheList.size();
      responses.resize(count);
      prefix.resize(count + 1);
      prefix[0] = 1.;
      for( std::size_t i = 0; i < count; ++i ){
        responses[i] = entry.dialResponseCacheList[i].getResponse();
        prefix[i+1] = prefix[i] * responses[i];
      }
      if( _globalEventReweightCap_.isEnabled and prefix.back() > _globalEventReweightCap_.maxReweight ){ continue; }
      double suffix = 1.;
      const double baseScale = scale * entry.event->getWeights().base;
      for( std::size_t i = count; i-- > 0; ){
        const double factor = baseScale * prefix[i] * suffix;
        suffix *= responses[i];
        const auto* interface = entry.dialResponseCacheList[i].dialInterface;
        const auto* input = interface->getInputBufferRef();
        if( input == nullptr or factor == 0. ){ continue; }
        for( int j = 0; j < input->getInputSize(); ++j ){
          const auto found = indices.find(&input->getParameter(j));
          if( found == indices.end() ){ continue; }
          sums[found->second] += factor * interface->evalGradient(j) * input->evalInputGradient(j);
        }
      }
    }
  });
  std::vector<double> result(parameterList_.size(), 0.);
  for( const auto& partial : partials ){
    for( std::size_t i = 0; i < result.size(); ++i ){ result[i] += partial[i]; }
  }
  return result;
}
