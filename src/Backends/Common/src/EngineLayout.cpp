#include "EngineLayout.h"

#include "CompactSpline.h"
#include "DialResponseSupervisor.h"
#include "EventDialCache.h"
#include "GeneralSpline.h"
#include "Graph.h"
#include "Histogram.h"
#include "LikelihoodInterface.h"
#include "MonotonicSpline.h"
#include "Norm.h"
#include "Parameter.h"
#include "Sample.h"
#include "SampleSet.h"
#include "Shift.h"
#include "UniformSpline.h"

#include <cmath>
#include <map>
#include <unordered_map>
#include <utility>

namespace {
  void buildEngineView(Backends::EngineView& view_, LikelihoodInterface& likelihoodInterface_) {
    auto& propagation = view_.propagation;
    auto& likelihood = view_.likelihood;

    auto& sampleSet = const_cast<Propagator&>(likelihoodInterface_.getModelPropagator()).getSampleSet();
    const auto& eventDialCache = likelihoodInterface_.getModelPropagator().getEventDialCache();

    propagation.samples.reserve(sampleSet.getSampleList().size());

    std::map<int, int> sampleBinOffsetMap;
    int binOffset{0};
    for( auto& sample : sampleSet.getSampleList() ){
      Backends::BackendSampleView sampleRef;
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
    std::unordered_map<const DialInterface*, Backends::BackendDialDescriptor> dialDescriptorMap{};

    for( const auto& cacheEntry : eventDialCache.getCache() ){
      if( cacheEntry.event == nullptr ){ continue; }
      if( cacheEntry.event->getIndices().bin < 0 ){ continue; }

      Backends::BackendEventView eventRef;
      eventRef.sampleIndex = cacheEntry.event->getIndices().sample;
      eventRef.binIndex = cacheEntry.event->getIndices().bin;
      eventRef.globalBinIndex = sampleBinOffsetMap.at(eventRef.sampleIndex) + eventRef.binIndex;
      eventRef.weight.baseWeight = cacheEntry.event->getWeights().base;
      eventRef.weight.firstDial = propagation.eventDials.size();
      eventRef.weight.dialCount = cacheEntry.dialResponseCacheList.size();
      eventRef.resultIndex = propagation.events.size();

      for( const auto& dialResponse : cacheEntry.dialResponseCacheList ){
        const auto* interface = dialResponse.dialInterface;
        LogThrowIf(interface == nullptr, "Null DialInterface while building EngineView.");
        auto cachedDescriptorIt = dialDescriptorMap.find(interface);
        if( cachedDescriptorIt != dialDescriptorMap.end() ){
          propagation.eventDials.emplace_back(cachedDescriptorIt->second);
          continue;
        }

        Backends::BackendDialDescriptor dialRef;
        const auto* dialBase = interface->getDialBaseRef();
        LogThrowIf(dialBase == nullptr, "Null DialBase while building EngineView.");
        const auto* inputBuffer = interface->getInputBufferRef();
        dialRef.firstInput = propagation.dialInputs.size();
        dialRef.inputCount = inputBuffer == nullptr ? 0 : std::size_t(inputBuffer->getBufferSize());
        dialRef.payloadOffset = propagation.dialPayloads.size();

        if( auto* supervisor = interface->getResponseSupervisorRef() ; supervisor != nullptr ){
          dialRef.hasMinResponse = not std::isnan(supervisor->getMinResponse());
          dialRef.hasMaxResponse = not std::isnan(supervisor->getMaxResponse());
          dialRef.minResponse = supervisor->getMinResponse();
          dialRef.maxResponse = supervisor->getMaxResponse();
        }

        if( dynamic_cast<const Norm*>(dialBase) != nullptr ){
          dialRef.type = Backends::BackendDialType::Norm;
        }
        else if( dynamic_cast<const Shift*>(dialBase) != nullptr ){
          dialRef.type = Backends::BackendDialType::Shift;
          propagation.dialPayloads.emplace_back(dialBase->evalResponse(DialInputBuffer()));
        }
        else if( dynamic_cast<const CompactSpline*>(dialBase) != nullptr ){
          dialRef.type = Backends::BackendDialType::CompactSpline;
          dialRef.allowExtrapolation = dialBase->getAllowExtrapolation();
          const auto& data = dialBase->getDialData();
          propagation.dialPayloads.insert(propagation.dialPayloads.end(), data.begin(), data.end());
        }
        else if( dynamic_cast<const UniformSpline*>(dialBase) != nullptr ){
          dialRef.type = Backends::BackendDialType::UniformSpline;
          dialRef.allowExtrapolation = dialBase->getAllowExtrapolation();
          const auto& data = dialBase->getDialData();
          propagation.dialPayloads.insert(propagation.dialPayloads.end(), data.begin(), data.end());
        }
        else if( dynamic_cast<const MonotonicSpline*>(dialBase) != nullptr ){
          dialRef.type = Backends::BackendDialType::MonotonicSpline;
          dialRef.allowExtrapolation = dialBase->getAllowExtrapolation();
          const auto& data = dialBase->getDialData();
          propagation.dialPayloads.insert(propagation.dialPayloads.end(), data.begin(), data.end());
        }
        else if( dynamic_cast<const GeneralSpline*>(dialBase) != nullptr ){
          dialRef.type = Backends::BackendDialType::GeneralSpline;
          dialRef.allowExtrapolation = dialBase->getAllowExtrapolation();
          const auto& data = dialBase->getDialData();
          propagation.dialPayloads.insert(propagation.dialPayloads.end(), data.begin(), data.end());
        }
        else if( dynamic_cast<const Graph*>(dialBase) != nullptr ){
          dialRef.type = Backends::BackendDialType::Graph;
          dialRef.allowExtrapolation = dialBase->getAllowExtrapolation();
          const auto& data = dialBase->getDialData();
          propagation.dialPayloads.insert(propagation.dialPayloads.end(), data.begin(), data.end());
        }
        else{
          LogThrow("Unsupported dial type for EngineView: " << dialBase->getDialTypeName());
        }

        dialRef.payloadSize = propagation.dialPayloads.size() - dialRef.payloadOffset;
        dialDescriptorMap.emplace(interface, dialRef);
        propagation.eventDials.emplace_back(dialRef);

        if( inputBuffer == nullptr ){ continue; }

        for( int iInput = 0 ; iInput < inputBuffer->getBufferSize() ; iInput++ ){
          const auto* parPtr = &inputBuffer->getParameter(iInput);
          auto parameterIndexIt = parameterIndexMap.find(parPtr);
          if( parameterIndexIt == parameterIndexMap.end() ){
            parameterIndexIt = parameterIndexMap.emplace(parPtr, parameterIndexMap.size()).first;
          }

          Backends::BackendDialInputDescriptor inputRef;
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

    propagation.parameterCount = parameterIndexMap.size();

    likelihood.samples.reserve(likelihoodInterface_.getSamplePairList().size());
    binOffset = 0;
    for( const auto& samplePair : likelihoodInterface_.getSamplePairList() ){
      Backends::BackendLikelihoodSampleView sampleRef;
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
}

void Backends::EngineLayout::clear() {
  view.clear();
  bindings.clear();
}

void Backends::EngineLayout::build(LikelihoodInterface& likelihoodInterface_) {
  clear();
  buildEngineView(view, likelihoodInterface_);
  bindings.build(likelihoodInterface_);
}
