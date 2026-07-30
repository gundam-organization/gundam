#include "BackendEngineView.h"

#include "EventDialCache.h"
#include "GenericToolbox.Utils.h"
#include "Histogram.h"
#include "LikelihoodInterface.h"
#include "Parameter.h"
#include "Sample.h"
#include "SampleSet.h"

#include <map>
#include <utility>

void Backends::BackendPropagationView::clear() {
  events.clear();
  eventDials.clear();
  inputBuffers.clear();
  samples.clear();
  parameters.clear();
  totalBins = 0;
}

void Backends::BackendEngineView::clear() {
  propagation.clear();
  likelihood.samples.clear();
}

void Backends::BackendEngineView::build(LikelihoodInterface& likelihoodInterface_) {
  clear();

  auto& sampleSet = const_cast<Propagator&>(likelihoodInterface_.getModelPropagator()).getSampleSet();
  const auto& eventDialCache = likelihoodInterface_.getModelPropagator().getEventDialCache();

  propagation.samples.reserve(sampleSet.getSampleList().size());

  std::map<int, int> sampleBinOffsetMap;
  int binOffset{0};
  for( auto& sample : sampleSet.getSampleList() ){
    BackendSampleRef sampleRef;
    sampleRef.histogram = &sample.getHistogram();
    sampleRef.sampleIndex = sample.getIndex();
    sampleRef.binOffset = binOffset;
    sampleRef.binCount = sample.getHistogram().getNbBins();
    propagation.samples.emplace_back(sampleRef);
    sampleBinOffsetMap[sampleRef.sampleIndex] = sampleRef.binOffset;
    binOffset += sampleRef.binCount;
  }
  propagation.totalBins = binOffset;

  propagation.events.reserve(eventDialCache.getCache().size());

  for( const auto& cacheEntry : eventDialCache.getCache() ){
    if( cacheEntry.event == nullptr ){ continue; }
    if( cacheEntry.event->getIndices().bin < 0 ){ continue; }

    BackendEventRef eventRef;
    eventRef.event = cacheEntry.event;
    eventRef.sampleIndex = cacheEntry.event->getIndices().sample;
    eventRef.binIndex = cacheEntry.event->getIndices().bin;
    eventRef.globalBinIndex = sampleBinOffsetMap.at(eventRef.sampleIndex) + eventRef.binIndex;
    eventRef.baseWeight = cacheEntry.event->getWeights().base;
    eventRef.firstDial = propagation.eventDials.size();
    eventRef.dialCount = cacheEntry.dialResponseCacheList.size();
    eventRef.resultIndex = propagation.events.size();

    for( const auto& dialResponse : cacheEntry.dialResponseCacheList ){
      BackendDialRef dialRef;
      dialRef.interface = dialResponse.dialInterface;
      propagation.eventDials.emplace_back(dialRef);

      const auto* inputBuffer = dialResponse.dialInterface->getInputBufferRef();
      GenericToolbox::addIfNotInVector(inputBuffer, propagation.inputBuffers);
      for( int iInput = 0 ; iInput < inputBuffer->getBufferSize() ; iInput++ ){
        const auto* parPtr = &inputBuffer->getParameter(iInput);
        GenericToolbox::addIfNotInVector(parPtr, propagation.parameters);
      }
    }

    propagation.events.emplace_back(eventRef);
  }

  likelihood.samples.reserve(likelihoodInterface_.getSamplePairList().size());
  binOffset = 0;
  for( const auto& samplePair : likelihoodInterface_.getSamplePairList() ){
    BackendLikelihoodSampleRef sampleRef;
    sampleRef.binOffset = binOffset;
    sampleRef.dataSums.reserve(samplePair.data->getHistogram().getNbBins());
    sampleRef.ignoredBins.reserve(samplePair.data->getHistogram().getNbBins());

    const auto& dataBinContentList = samplePair.data->getHistogram().getBinContentList();
    const auto& modelBinContentList = samplePair.model->getHistogram().getBinContentList();
    for( int iBin = 0 ; iBin < samplePair.data->getHistogram().getNbBins() ; iBin++ ){
      sampleRef.dataSums.emplace_back(dataBinContentList[iBin].sumWeights);
      sampleRef.ignoredBins.emplace_back(
          likelihoodInterface_.getJointProbabilityPtr()->isIgnoreBinsWithZeroPredictionAtPrior()
          and modelBinContentList[iBin].sumWeights == 0.
      );
    }

    auto* jointProbability = likelihoodInterface_.getJointProbabilityPtr();
    sampleRef.evalBin = [jointProbability](double data_, double pred_, double err_, int bin_){
      return jointProbability->eval(data_, pred_, err_, bin_);
    };

    likelihood.samples.emplace_back(std::move(sampleRef));
    binOffset += samplePair.model->getHistogram().getNbBins();
  }
}
