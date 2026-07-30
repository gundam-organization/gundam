#include "BackendEngineView.h"

#include "EventDialCache.h"
#include "Histogram.h"
#include "LikelihoodInterface.h"
#include "Parameter.h"
#include "Sample.h"
#include "SampleSet.h"

#include <cmath>
#include <map>
#include <unordered_map>
#include <utility>

void Backends::BackendPropagationView::clear() {
  events.clear();
  eventDials.clear();
  dialInputs.clear();
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
    sampleRef.sampleIndex = sample.getIndex();
    sampleRef.binOffset = binOffset;
    sampleRef.binCount = sample.getHistogram().getNbBins();
    propagation.samples.emplace_back(sampleRef);
    sampleBinOffsetMap[sampleRef.sampleIndex] = sampleRef.binOffset;
    binOffset += sampleRef.binCount;
  }
  propagation.totalBins = binOffset;

  propagation.events.reserve(eventDialCache.getCache().size());
  std::unordered_map<const Parameter*, std::size_t> parameterIndexMap{};

  for( const auto& cacheEntry : eventDialCache.getCache() ){
    if( cacheEntry.event == nullptr ){ continue; }
    if( cacheEntry.event->getIndices().bin < 0 ){ continue; }

    BackendEventRef eventRef;
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
      const auto* inputBuffer = dialResponse.dialInterface->getInputBufferRef();
      dialRef.firstInput = propagation.dialInputs.size();
      dialRef.inputCount = inputBuffer == nullptr ? 0 : std::size_t(inputBuffer->getBufferSize());
      propagation.eventDials.emplace_back(dialRef);

      if( inputBuffer == nullptr ){ continue; }

      for( int iInput = 0 ; iInput < inputBuffer->getBufferSize() ; iInput++ ){
        const auto* parPtr = &inputBuffer->getParameter(iInput);
        auto parameterIndexIt = parameterIndexMap.find(parPtr);
        if( parameterIndexIt == parameterIndexMap.end() ){
          propagation.parameters.emplace_back(parPtr);
          parameterIndexIt = parameterIndexMap.emplace(parPtr, propagation.parameters.size() - 1).first;
        }

        BackendDialInputRef inputRef;
        inputRef.parameterIndex = parameterIndexIt->second;
        const auto& mirrorEdges = inputBuffer->getMirrorEdges(iInput);
        inputRef.useMirror = not std::isnan(mirrorEdges.minValue);
        inputRef.mirrorMin = mirrorEdges.minValue;
        inputRef.mirrorRange = mirrorEdges.range;
        propagation.dialInputs.emplace_back(inputRef);
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
