#include "BackendModelBuilder.h"

#include "DialInputBuffer.h"
#include "DialInterface.h"
#include "Event.h"
#include "EventDialCache.h"
#include "Histogram.h"
#include "Parameter.h"
#include "Sample.h"
#include "SampleSet.h"

#include "GenericToolbox.Utils.h"

#include <map>

Backends::BackendEngineView Backends::BackendEngineViewBuilder::build(
    SampleSet& sampleSet_,
    const EventDialCache& eventDialCache_) {

  BackendEngineView engineView;
  auto& model = engineView.propagation;
  model.samples.reserve(sampleSet_.getSampleList().size());

  std::map<int, int> sampleBinOffsetMap;
  int binOffset{0};
  for( auto& sample : sampleSet_.getSampleList() ){
    BackendSampleRef sampleRef;
    sampleRef.histogram = &sample.getHistogram();
    sampleRef.sampleIndex = sample.getIndex();
    sampleRef.binOffset = binOffset;
    sampleRef.binCount = sample.getHistogram().getNbBins();
    model.samples.emplace_back(sampleRef);
    sampleBinOffsetMap[sampleRef.sampleIndex] = sampleRef.binOffset;
    binOffset += sampleRef.binCount;
  }
  model.totalBins = binOffset;

  model.events.reserve(eventDialCache_.getCache().size());

  for( const auto& cacheEntry : eventDialCache_.getCache() ){
    if( cacheEntry.event == nullptr ){ continue; }
    if( cacheEntry.event->getIndices().bin < 0 ){ continue; }

    BackendEventRef eventRef;
    eventRef.event = cacheEntry.event;
    eventRef.sampleIndex = cacheEntry.event->getIndices().sample;
    eventRef.binIndex = cacheEntry.event->getIndices().bin;
    eventRef.globalBinIndex = sampleBinOffsetMap.at(eventRef.sampleIndex) + eventRef.binIndex;
    eventRef.baseWeight = cacheEntry.event->getWeights().base;
    eventRef.firstDial = model.eventDials.size();
    eventRef.dialCount = cacheEntry.dialResponseCacheList.size();
    eventRef.resultIndex = model.events.size();

    for( const auto& dialResponse : cacheEntry.dialResponseCacheList ){
      BackendDialRef dialRef;
      dialRef.interface = dialResponse.dialInterface;
      model.eventDials.emplace_back(dialRef);

      const auto* inputBuffer = dialResponse.dialInterface->getInputBufferRef();
      GenericToolbox::addIfNotInVector(inputBuffer, model.inputBuffers);
      for( int iInput = 0 ; iInput < inputBuffer->getBufferSize() ; iInput++ ){
        const auto* parPtr = &inputBuffer->getParameter(iInput);
        GenericToolbox::addIfNotInVector(parPtr, model.parameters);
      }
    }

    model.events.emplace_back(eventRef);
  }

  return engineView;
}
