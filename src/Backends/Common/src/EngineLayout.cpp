#include "EngineLayout.h"

#include "Bilinear.h"
#include "Bicubic.h"
#include "CompactSpline.h"
#include "DialResponseSupervisor.h"
#include "EventDialCache.h"
#include "ExternalWeightDispatcher.h"
#include "Tabulated.h"
#include "Kriged.h"
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

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {
  void buildEngineView(Backends::EngineView& view_, Backends::EngineBindings& bindings_, LikelihoodInterface& likelihoodInterface_) {
    auto& propagation = view_.propagation;
    auto& likelihood = view_.likelihood;

    auto& sampleSet = const_cast<Propagator&>(likelihoodInterface_.getModelPropagator()).getSampleSet();
    const auto& eventDialCache = likelihoodInterface_.getModelPropagator().getEventDialCache();

    propagation.samples.reserve(sampleSet.getSampleList().size());

    std::map<int, std::size_t> samplePositionMap;
    int binOffset{0};
    for( auto& sample : sampleSet.getSampleList() ){
      Backends::SampleView sampleRef;
      sampleRef.sampleIndex = sample.getIndex();
      sampleRef.binOffset = binOffset;
      sampleRef.binCount = sample.getHistogram().getNbBins();
      sampleRef.useDynamicBinning = sample.isUseDynamicBinning();
      sampleRef.readHostVariables = sample.isUseDynamicBinning() and eventDialCache.hasVariableResponses()
          and not likelihoodInterface_.getModelPropagator().getEventVariableResponseCache().isAutomaticBinningSample(propagation.samples.size());
      if( sampleRef.useDynamicBinning or sample.getHistogram().isBinningResponseEnabled() ){
        sampleRef.binning = Backends::buildBinningGeometry(sample.getHistogram());
        sampleRef.variableNames = sampleRef.binning.variableNames;
      }
      samplePositionMap[sampleRef.sampleIndex] = propagation.samples.size();
      propagation.samples.emplace_back(std::move(sampleRef));
      binOffset += sample.getHistogram().getNbBins();
    }
    propagation.totalBins = binOffset;
    propagation.hasVariableResponses = eventDialCache.hasVariableResponses();
    propagation.hasReweightCap = eventDialCache.getGlobalEventReweightCap().isEnabled;
    propagation.maxReweight = eventDialCache.getGlobalEventReweightCap().maxReweight;
    // Every event in a sample uses the same compact variable schema. Binning
    // coordinates come first; auxiliary dial targets follow.
    for( const auto& entry : eventDialCache.getCache() ){
      if( entry.event == nullptr ){ continue; }
      auto& sample = propagation.samples.at(samplePositionMap.at(entry.event->getIndices().sample));
      for( const auto& response : entry.variableResponseCacheList ){
        const auto& name = entry.event->getVariables().getNameListPtr()->at(response.variableIndex);
        if( std::find(sample.variableNames.begin(), sample.variableNames.end(), name) == sample.variableNames.end() ){
          sample.variableNames.emplace_back(name);
        }
      }
    }

    propagation.events.reserve(eventDialCache.getCache().size());
    propagation.eventWeights.reserve(eventDialCache.getCache().size());
    std::unordered_map<const Parameter*, std::size_t> parameterIndexMap{};
    for( std::size_t i = 0; i < bindings_.parameters.size(); ++i ){
      parameterIndexMap.emplace(bindings_.parameters[i].parameter, i);
    }
    std::unordered_map<const DialInterface*, std::uint32_t> dialIndexMap{};
    // Count the flat buffers before filling them. Reuse this map during
    // compilation, preserving the first-encounter order of the dial indices.
    // This avoids geometric vector growth and its temporary payload copies.
    constexpr auto uncompiledDial = std::numeric_limits<std::uint32_t>::max();
    std::size_t payloadCount{0};
    std::size_t eventDialCount{0};
    for( const auto& entry : eventDialCache.getCache() ){
      if( entry.event == nullptr ){ continue; }
      for( const auto& response : entry.dialResponseCacheList ){
        if( not response.appliesToVariable ){ ++eventDialCount; }
        const auto* interface = response.dialInterface;
        LogThrowIf(interface == nullptr, "Null DialInterface while building EngineView.");
        if( not dialIndexMap.emplace(interface, uncompiledDial).second ){ continue; }
        const auto* dial = interface->getDialBaseRef();
        LogThrowIf(dial == nullptr, "Null DialBase while building EngineView.");
        if( dynamic_cast<const Shift*>(dial) != nullptr ){ ++payloadCount; }
        else if( dynamic_cast<const Norm*>(dial) == nullptr
                 and dynamic_cast<const ExternalWeightDispatcher*>(dial) == nullptr
                 and dynamic_cast<const Tabulated*>(dial) == nullptr
                 and dynamic_cast<const Kriged*>(dial) == nullptr ){
          payloadCount += dial->getDialData().size();
        }
      }
    }
    LogThrowIf(dialIndexMap.size() >= uncompiledDial, "Too many dials for EngineView indices.");
    propagation.dials.reserve(dialIndexMap.size());
    propagation.dialPayloads.reserve(payloadCount);
    propagation.eventDialIndices.reserve(eventDialCount);
    // Dial collections share their input buffers. Keep each input sequence
    // once, including its parameter order and mirror settings.
    std::unordered_map<const DialInputBuffer*, std::size_t> inputOffsetMap{};
    std::unordered_map<const ExternalWeightBuffer*, std::size_t> externalSourceIndices{};
    std::unordered_map<const std::vector<double>*, std::size_t> tableSourceIndices{};
    std::size_t externalWeightCount{0};
    std::unordered_set<const DialInputBuffer*> variableInputBuffers;

    for( const auto& cacheEntry : eventDialCache.getCache() ){
      if( cacheEntry.event == nullptr ){ continue; }

      Backends::EventView eventRef;
      Backends::BackendEventWeightDescriptor weightRef;
      eventRef.sampleIndex = cacheEntry.event->getIndices().sample;
      eventRef.binIndex = cacheEntry.event->getIndices().bin;
      eventRef.samplePosition = samplePositionMap.at(eventRef.sampleIndex);
      const auto& sample = propagation.samples.at(eventRef.samplePosition);
      eventRef.globalBinIndex = eventRef.binIndex < 0 ? -1 : sample.binOffset + eventRef.binIndex;
      weightRef.baseWeight = cacheEntry.event->getWeights().base;
      weightRef.firstDial = propagation.eventDialIndices.size();

      eventRef.resultIndex = propagation.events.size();

      std::vector<std::uint32_t> allDialIndices;
      for( const auto& dialResponse : cacheEntry.dialResponseCacheList ){
        const auto* interface = dialResponse.dialInterface;
        LogThrowIf(interface == nullptr, "Null DialInterface while building EngineView.");
        if( dialResponse.appliesToVariable ){
          const auto* buffer = interface->getInputBufferRef();
          if( buffer != nullptr and variableInputBuffers.insert(buffer).second ){
            bindings_.variableInputBuffers.emplace_back(buffer);
          }
        }
        auto cachedDialIt = dialIndexMap.find(interface);
        if( cachedDialIt->second != uncompiledDial ){
          allDialIndices.emplace_back(cachedDialIt->second);
          continue;
        }

        Backends::BackendDialDescriptor dialRef;
        const auto* dialBase = interface->getDialBaseRef();
        LogThrowIf(dialBase == nullptr, "Null DialBase while building EngineView.");
        const auto* inputBuffer = interface->getInputBufferRef();
        auto inputOffset = inputOffsetMap.emplace(inputBuffer, propagation.dialInputs.size());
        dialRef.firstInput = inputOffset.first->second;
        dialRef.inputCount = inputBuffer == nullptr ? 0 : std::size_t(inputBuffer->getBufferSize());
        dialRef.payloadOffset = propagation.dialPayloads.size();

        if( auto* supervisor = interface->getResponseSupervisorRef() ; supervisor != nullptr ){
          dialRef.hasMinResponse = not std::isnan(supervisor->getMinResponse());
          dialRef.hasMaxResponse = not std::isnan(supervisor->getMaxResponse());
          auto& smoothing = dialRef.smoothing;
          smoothing.minResponse = supervisor->getMinResponse();
          smoothing.maxResponse = supervisor->getMaxResponse();
          smoothing.minWidth = supervisor->getMinSmoothingWidth();
          smoothing.maxWidth = supervisor->getMaxSmoothingWidth();
          smoothing.minInverseWidth = smoothing.minWidth > 0. ? 0.5 / smoothing.minWidth : 0.;
          smoothing.maxInverseWidth = smoothing.maxWidth > 0. ? 0.5 / smoothing.maxWidth : 0.;
          dialRef.minResponse = smoothing.minResponse + smoothing.minWidth;
          dialRef.maxResponse = smoothing.maxResponse - smoothing.maxWidth;
        }

        if( auto* external = dynamic_cast<const ExternalWeightDispatcher*>(dialBase) ){
          const auto& source = external->getWeightSource();
          LogThrowIf(source == nullptr or external->getWeightIndex() >= source->size(),
                     "Invalid external weight source while building EngineView.");
          auto inserted = externalSourceIndices.emplace(source.get(), bindings_.externalWeights.size());
          if( inserted.second ){
            bindings_.externalWeights.push_back({source});
            propagation.externalWeightBlocks.push_back({externalWeightCount, source->size()});
            externalWeightCount += source->size();
          }
          dialRef.type = Backends::BackendDialType::ExternalWeight;
          dialRef.externalBlockIndex = inserted.first->second;
          dialRef.externalWeightIndex = external->getWeightIndex();
        }
        else if( dynamic_cast<const Tabulated*>(dialBase) != nullptr
                 or dynamic_cast<const Kriged*>(dialBase) != nullptr ){
          const auto* tabulated = dynamic_cast<const Tabulated*>(dialBase);
          const auto* kriged = dynamic_cast<const Kriged*>(dialBase);
          const auto* table = tabulated != nullptr ? tabulated->getTable() : kriged->getTable();
          LogThrowIf(table == nullptr, "Null table while building EngineView.");
          auto inserted = tableSourceIndices.emplace(table, bindings_.externalWeights.size());
          if( inserted.second ){
            bindings_.externalWeights.push_back({nullptr, table});
            propagation.externalWeightBlocks.push_back({externalWeightCount, table->size()});
            externalWeightCount += table->size();
          }
          dialRef.externalBlockIndex = inserted.first->second;
          dialRef.firstTableTerm = propagation.tableTerms.size();
          if( tabulated != nullptr ){
            dialRef.type = Backends::BackendDialType::Tabulated;
            LogThrowIf(tabulated->getIndex() < 0 or table->size() < 2
                       or std::size_t(tabulated->getIndex()) >= table->size() - 1,
                       "Invalid Tabulated table index.");
            const auto index = std::size_t(tabulated->getIndex());
            propagation.tableTerms.push_back({index, tabulated->getFraction()});
            propagation.tableTerms.push_back({index + 1, 1. - tabulated->getFraction()});
          }
          else{
            dialRef.type = Backends::BackendDialType::Kriged;
            for( const auto& weight : kriged->getWeights() ){
              LogThrowIf(weight.first < 0 or std::size_t(weight.first) >= table->size(),
                         "Invalid Kriged table index.");
              propagation.tableTerms.push_back({std::size_t(weight.first), weight.second});
            }
          }
          dialRef.tableTermCount = propagation.tableTerms.size() - dialRef.firstTableTerm;
        }
        else if( dynamic_cast<const Norm*>(dialBase) != nullptr ){
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
        else if( dynamic_cast<const Bilinear*>(dialBase) != nullptr or dynamic_cast<const Bicubic*>(dialBase) != nullptr ){
          dialRef.type = dynamic_cast<const Bilinear*>(dialBase) != nullptr
              ? Backends::BackendDialType::Bilinear : Backends::BackendDialType::Bicubic;
          LogThrowIf(dialRef.inputCount != 2, "Surface dials require exactly two inputs.");
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
        const auto dialIndex = std::uint32_t(propagation.dials.size());
        propagation.dials.emplace_back(dialRef);
        cachedDialIt->second = dialIndex;
        allDialIndices.emplace_back(dialIndex);

        if( inputBuffer == nullptr or not inputOffset.second ){ continue; }

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

      for( std::size_t i = 0; i < allDialIndices.size(); ++i ){
        if( not cacheEntry.dialResponseCacheList[i].appliesToVariable ){
          propagation.eventDialIndices.emplace_back(allDialIndices[i]);
        }
      }
      weightRef.dialCount = propagation.eventDialIndices.size() - weightRef.firstDial;
      eventRef.firstVariable = propagation.variableValues.size();
      eventRef.variableCount = sample.variableNames.size();
      const auto& variables = cacheEntry.event->getVariables();
      for( const auto& name : sample.variableNames ){
        int index = variables.findVarIndex(name);
        bool isTarget = std::any_of(cacheEntry.variableResponseCacheList.begin(), cacheEntry.variableResponseCacheList.end(),
                                    [&](const auto& target_){ return target_.variableIndex == index; });
        bindings_.variables.push_back({cacheEntry.event, index, propagation.variableValues.size(), sample.readHostVariables, isTarget});
        propagation.variableValues.emplace_back(variables.getVarList()[index].getVarAsDouble());
      }
      eventRef.firstVariableResponse = propagation.variableResponses.size();
      for( const auto& target : cacheEntry.variableResponseCacheList ){
        Backends::BackendVariableResponseDescriptor response;
        const auto& name = variables.getNameListPtr()->at(target.variableIndex);
        response.variableIndex = eventRef.firstVariable
            + std::size_t(std::find(sample.variableNames.begin(), sample.variableNames.end(), name) - sample.variableNames.begin());
        response.referenceValue = target.referenceValue;
        if( target.overrideDial != std::size_t(-1) ){ response.overrideDial = allDialIndices.at(target.overrideDial); }
        response.firstMultiplyDial = propagation.variableDialIndices.size();
        for( auto dial : target.multiplyDials ){ propagation.variableDialIndices.emplace_back(allDialIndices.at(dial)); }
        response.multiplyDialCount = propagation.variableDialIndices.size() - response.firstMultiplyDial;
        response.firstAddDial = propagation.variableDialIndices.size();
        for( auto dial : target.addDials ){ propagation.variableDialIndices.emplace_back(allDialIndices.at(dial)); }
        response.addDialCount = propagation.variableDialIndices.size() - response.firstAddDial;
        propagation.variableResponses.emplace_back(response);
      }
      eventRef.variableResponseCount = propagation.variableResponses.size() - eventRef.firstVariableResponse;
      propagation.events.emplace_back(eventRef);
      propagation.eventWeights.emplace_back(weightRef);
    }

    propagation.parameterCount = parameterIndexMap.size();
    propagation.variableInputBufferCount = bindings_.variableInputBuffers.size();

    likelihood.samples.reserve(likelihoodInterface_.getSamplePairList().size());
    binOffset = 0;
    for( const auto& samplePair : likelihoodInterface_.getSamplePairList() ){
      Backends::LikelihoodSampleView sampleRef;
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
      sampleRef.evalBin = [jointProbability, samplePair](double data_, double pred_, double err_, int bin_){
        return jointProbability->evalWithPrediction(samplePair, data_, pred_, err_, bin_);
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
  bindings.build(likelihoodInterface_);
  buildEngineView(view, bindings, likelihoodInterface_);
}
