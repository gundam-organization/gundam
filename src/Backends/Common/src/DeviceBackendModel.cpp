#include "DeviceBackendModel.h"

#include "Logger.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <unordered_map>
#include <utility>

namespace {

  [[nodiscard]] double secondsSince(std::chrono::steady_clock::time_point start_) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  }

}

void Backends::DevicePackedModel::clear() {
  *this = DevicePackedModel();
}

std::uint32_t Backends::DevicePackedModel::uniqueDialCount() const {
  return compactDialDescriptorCount + uniformDialDescriptorCount
         + monotonicDialDescriptorCount + generalDialDescriptorCount
         + graphDialDescriptorCount;
}

bool Backends::packDeviceBackendModel(const PropagationView& model_,
                                      bool enableCachedDialResponses_,
                                      DevicePackedModel& packedModel_,
                                      BackendTimingSummary& buildTiming_,
                                      std::string& fallbackReason_,
                                      const std::string& logPrefix_) {
  packedModel_.clear();
  fallbackReason_.clear();
  auto buildStart = std::chrono::steady_clock::now();
  auto lastStageStart = buildStart;

  auto fail = [&fallbackReason_](std::string reason_) {
    fallbackReason_ = std::move(reason_);
    return false;
  };

  if( model_.events.empty() ){
    return fail("the backend model has no events.");
  }
  if( model_.totalBins <= 0 ){
    return fail("the backend model has no histogram bins.");
  }

  LogInfo << logPrefix_ << ": building packed device model for "
          << model_.events.size() << " events, "
          << model_.eventDialIndices.size() << " event dials, "
          << model_.parameterCount << " parameters and "
          << model_.totalBins << " histogram bins."
          << std::endl;

  LogInfo << logPrefix_ << ": compatibility scan done in "
          << secondsSince(lastStageStart) << " s."
          << std::endl;
  buildTiming_.buildCompatibilityScanSeconds = secondsSince(lastStageStart);
  lastStageStart = std::chrono::steady_clock::now();
  LogInfo << logPrefix_ << ": built parameter lookup table in "
          << secondsSince(lastStageStart) << " s."
          << std::endl;
  buildTiming_.buildParameterLookupSeconds = secondsSince(lastStageStart);
  lastStageStart = std::chrono::steady_clock::now();

  packedModel_.baseWeights.resize(model_.events.size());
  packedModel_.eventDialRanges.resize(model_.events.size());
  packedModel_.globalBins.resize(model_.events.size());
  std::vector<std::uint32_t> eventsPerBin(model_.totalBins, 0);
  std::size_t shiftCount{0};
  std::size_t normCount{0};
  std::size_t compactSplineCount{0};
  std::size_t uniformSplineCount{0};
  std::size_t monotonicSplineCount{0};
  std::size_t generalSplineCount{0};
  std::size_t graphCount{0};
  std::size_t uniqueSplineScalarCount{0};
  std::vector<std::size_t> uniqueCompactDialOffsets{};
  std::vector<std::size_t> uniqueUniformDialOffsets{};
  std::vector<std::size_t> uniqueMonotonicDialOffsets{};
  std::vector<std::size_t> uniqueGeneralDialOffsets{};
  std::vector<std::size_t> uniqueGraphDialOffsets{};
  std::vector<std::uint32_t> compactDialReuseCounts{};
  std::vector<std::uint32_t> uniformDialReuseCounts{};
  std::vector<std::uint32_t> monotonicDialReuseCounts{};
  std::vector<std::uint32_t> generalDialReuseCounts{};
  std::vector<std::uint32_t> graphDialReuseCounts{};
  uniqueCompactDialOffsets.reserve(model_.dials.size());
  uniqueUniformDialOffsets.reserve(model_.dials.size());
  uniqueMonotonicDialOffsets.reserve(model_.dials.size());
  uniqueGeneralDialOffsets.reserve(model_.dials.size());
  uniqueGraphDialOffsets.reserve(model_.dials.size());
  compactDialReuseCounts.reserve(model_.dials.size());
  uniformDialReuseCounts.reserve(model_.dials.size());
  monotonicDialReuseCounts.reserve(model_.dials.size());
  generalDialReuseCounts.reserve(model_.dials.size());
  graphDialReuseCounts.reserve(model_.dials.size());
  std::unordered_map<std::size_t, DevicePackedDialRef> packedDialIndexMap{};
  packedDialIndexMap.reserve(model_.dials.size());
  std::unordered_map<std::size_t, const BackendDialDescriptor*> packedDialDescriptorMap{};
  packedDialDescriptorMap.reserve(model_.dials.size());

  LogInfo << logPrefix_ << ": first packing pass."
          << " This phase inventories unique shared dials and precomputes payload sizes."
          << std::endl;

  constexpr std::size_t kPackingProgressEventStep = 10000;
  auto packingLoopStart = std::chrono::steady_clock::now();
  auto lastPackingProgress = packingLoopStart;
  std::size_t processedDialRefs{0};
  for( std::size_t iEvent = 0 ; iEvent < model_.events.size() ; iEvent++ ){
    const auto& event = model_.events[iEvent];
    if( event.globalBinIndex < 0 or event.globalBinIndex >= model_.totalBins ){
      return fail("at least one event has an invalid global bin index.");
    }
    for( std::size_t iDial = 0 ; iDial < event.weight.dialCount ; iDial++ ){
      processedDialRefs++;
      const auto& eventDial = model_.dials[model_.eventDialIndices[event.weight.firstDial + iDial]];
      if( eventDial.type == BackendDialType::Shift ){
        shiftCount++;
        continue;
      }

      if( eventDial.inputCount != 1 ){
        return fail("at least one backend dial is not device-compatible because it does not have exactly one input parameter.");
      }

      if( eventDial.type == BackendDialType::Norm ){
        normCount++;
        packedModel_.totalDynamicDialOccurrences++;
        continue;
      }

      auto packedDialIndexIt = packedDialIndexMap.find(eventDial.payloadOffset);
      if( packedDialIndexIt != packedDialIndexMap.end() ){
        auto packedDialRef = packedDialIndexIt->second;
        switch( packedDialRef.type ){
          case kDeviceDialTypeCompactSpline: compactDialReuseCounts[packedDialRef.localIndex]++; break;
          case kDeviceDialTypeUniformSpline: uniformDialReuseCounts[packedDialRef.localIndex]++; break;
          case kDeviceDialTypeMonotonicSpline: monotonicDialReuseCounts[packedDialRef.localIndex]++; break;
          case kDeviceDialTypeGeneralSpline: generalDialReuseCounts[packedDialRef.localIndex]++; break;
          case kDeviceDialTypeGraph: graphDialReuseCounts[packedDialRef.localIndex]++; break;
          default: LogThrow("Internal device packing error: unexpected dial type in reuse table.");
        }
        packedModel_.totalDynamicDialOccurrences++;
        continue;
      }

      DevicePackedDialRef packedDialRef;
      packedModel_.totalDynamicDialOccurrences++;
      if( eventDial.type == BackendDialType::CompactSpline ){
        if( eventDial.payloadSize < 6 ){
          return fail("CompactSpline dial data is too small for device evaluation.");
        }
        uniqueSplineScalarCount += eventDial.payloadSize;
        compactSplineCount++;
        packedDialRef.type = kDeviceDialTypeCompactSpline;
        packedDialRef.localIndex = std::uint32_t(uniqueCompactDialOffsets.size());
        uniqueCompactDialOffsets.emplace_back(eventDial.payloadOffset);
        compactDialReuseCounts.emplace_back(1);
      }
      else if( eventDial.type == BackendDialType::UniformSpline ){
        if( eventDial.payloadSize < 8 ){
          return fail("UniformSpline dial data is too small for device evaluation.");
        }
        uniqueSplineScalarCount += eventDial.payloadSize;
        uniformSplineCount++;
        packedDialRef.type = kDeviceDialTypeUniformSpline;
        packedDialRef.localIndex = std::uint32_t(uniqueUniformDialOffsets.size());
        uniqueUniformDialOffsets.emplace_back(eventDial.payloadOffset);
        uniformDialReuseCounts.emplace_back(1);
      }
      else if( eventDial.type == BackendDialType::MonotonicSpline ){
        if( eventDial.payloadSize < 5 ){
          return fail("MonotonicSpline dial data is too small for device evaluation.");
        }
        uniqueSplineScalarCount += eventDial.payloadSize;
        monotonicSplineCount++;
        packedDialRef.type = kDeviceDialTypeMonotonicSpline;
        packedDialRef.localIndex = std::uint32_t(uniqueMonotonicDialOffsets.size());
        uniqueMonotonicDialOffsets.emplace_back(eventDial.payloadOffset);
        monotonicDialReuseCounts.emplace_back(1);
      }
      else if( eventDial.type == BackendDialType::GeneralSpline ){
        if( eventDial.payloadSize < 11 ){
          return fail("GeneralSpline dial data is too small for device evaluation.");
        }
        uniqueSplineScalarCount += eventDial.payloadSize;
        generalSplineCount++;
        packedDialRef.type = kDeviceDialTypeGeneralSpline;
        packedDialRef.localIndex = std::uint32_t(uniqueGeneralDialOffsets.size());
        uniqueGeneralDialOffsets.emplace_back(eventDial.payloadOffset);
        generalDialReuseCounts.emplace_back(1);
      }
      else if( eventDial.type == BackendDialType::Graph ){
        if( eventDial.payloadSize < 2 ){
          return fail("Graph dial data is too small for device evaluation.");
        }
        uniqueSplineScalarCount += eventDial.payloadSize;
        graphCount++;
        packedDialRef.type = kDeviceDialTypeGraph;
        packedDialRef.localIndex = std::uint32_t(uniqueGraphDialOffsets.size());
        uniqueGraphDialOffsets.emplace_back(eventDial.payloadOffset);
        graphDialReuseCounts.emplace_back(1);
      }
      else{
        return fail("unexpected backend dial type has no device encoder.");
      }
      packedDialIndexMap.emplace(eventDial.payloadOffset, packedDialRef);
      packedDialDescriptorMap.emplace(eventDial.payloadOffset, &eventDial);
    }

    if( ((iEvent + 1) % kPackingProgressEventStep) == 0 or (iEvent + 1) == model_.events.size() ){
      auto now = std::chrono::steady_clock::now();
      LogInfo << logPrefix_ << ": first pass progress "
              << (iEvent + 1) << "/" << model_.events.size()
              << " events, "
              << processedDialRefs << "/" << model_.eventDialIndices.size()
              << " dial refs scanned, "
              << packedDialIndexMap.size() << " unique dynamic dials found, elapsed "
              << secondsSince(packingLoopStart) << " s"
              << " (+" << std::chrono::duration<double>(now - lastPackingProgress).count() << " s)"
              << "."
              << std::endl;
      lastPackingProgress = now;
    }
  }
  LogInfo << logPrefix_ << ": first pass completed in "
          << secondsSince(lastStageStart) << " s"
          << " [unique dynamic dials=" << packedDialIndexMap.size()
          << ", unique spline scalars=" << uniqueSplineScalarCount
          << "]."
          << std::endl;
  buildTiming_.buildFirstPassSeconds = secondsSince(lastStageStart);
  lastStageStart = std::chrono::steady_clock::now();

  packedModel_.compactDialDescriptors.reserve(uniqueCompactDialOffsets.size());
  packedModel_.uniformDialDescriptors.reserve(uniqueUniformDialOffsets.size());
  packedModel_.monotonicDialDescriptors.reserve(uniqueMonotonicDialOffsets.size());
  packedModel_.generalDialDescriptors.reserve(uniqueGeneralDialOffsets.size());
  packedModel_.graphDialDescriptors.reserve(uniqueGraphDialOffsets.size());
  packedModel_.normDialOccurrences.reserve(normCount);
  packedModel_.compactDialIndices.reserve(compactSplineCount);
  packedModel_.uniformDialIndices.reserve(uniformSplineCount);
  packedModel_.monotonicDialIndices.reserve(monotonicSplineCount);
  packedModel_.generalDialIndices.reserve(generalSplineCount);
  packedModel_.graphDialIndices.reserve(graphCount);
  packedModel_.splineData.reserve(uniqueSplineScalarCount);

  LogInfo << logPrefix_ << ": second packing pass."
          << " This phase materializes unique dial descriptors and payloads."
          << std::endl;
  packedModel_.cachedDialCount = 0;
  auto fillMinMax = [](const BackendDialDescriptor& dialRef_, float& minResponse_, float& maxResponse_) {
    minResponse_ = -std::numeric_limits<float>::infinity();
    maxResponse_ = std::numeric_limits<float>::infinity();
    if( dialRef_.hasMinResponse ){ minResponse_ = float(dialRef_.minResponse); }
    if( dialRef_.hasMaxResponse ){ maxResponse_ = float(dialRef_.maxResponse); }
  };
  auto packDescriptors = [&](const std::vector<std::size_t>& dialOffsets_,
                             const std::vector<std::uint32_t>& reuseCounts_,
                             std::vector<DeviceSplineDialDescriptor>& descriptors_,
                             std::uint32_t& cachedCount_) {
    for( std::size_t iUniqueDial = 0 ; iUniqueDial < dialOffsets_.size() ; iUniqueDial++ ){
      auto eventDialIt = packedDialDescriptorMap.find(dialOffsets_[iUniqueDial]);
      LogThrowIf(eventDialIt == packedDialDescriptorMap.end(),
                 "Internal device packing error: could not resolve unique backend dial descriptor.");
      const auto& eventDial = *eventDialIt->second;

      DeviceSplineDialDescriptor descriptor;
      descriptor.parameterIndex = std::uint32_t(model_.dialInputs.at(eventDial.firstInput).parameterIndex);
      descriptor.splineOffset = std::uint32_t(packedModel_.splineData.size());
      descriptor.splineSize = std::uint32_t(eventDial.payloadSize);
      descriptor.flags = eventDial.allowExtrapolation ? kDeviceDialFlagAllowExtrapolation : 0u;
      fillMinMax(eventDial, descriptor.minResponse, descriptor.maxResponse);

      if( enableCachedDialResponses_ and reuseCounts_[iUniqueDial] >= kDeviceCachedDialReuseThreshold ){
        descriptor.flags |= kDeviceDialFlagCached;
        cachedCount_++;
        packedModel_.cachedDialCount++;
      }

      for( std::size_t iPayload = 0 ; iPayload < eventDial.payloadSize ; iPayload++ ){
        packedModel_.splineData.emplace_back(float(model_.dialPayloads.at(eventDial.payloadOffset + iPayload)));
      }
      descriptors_.emplace_back(descriptor);
    }
  };

  packedModel_.compactCachedDialCount = 0;
  packedModel_.uniformCachedDialCount = 0;
  packedModel_.monotonicCachedDialCount = 0;
  packedModel_.generalCachedDialCount = 0;
  packedModel_.graphCachedDialCount = 0;
  packDescriptors(uniqueCompactDialOffsets, compactDialReuseCounts, packedModel_.compactDialDescriptors, packedModel_.compactCachedDialCount);
  packDescriptors(uniqueUniformDialOffsets, uniformDialReuseCounts, packedModel_.uniformDialDescriptors, packedModel_.uniformCachedDialCount);
  packDescriptors(uniqueMonotonicDialOffsets, monotonicDialReuseCounts, packedModel_.monotonicDialDescriptors, packedModel_.monotonicCachedDialCount);
  packDescriptors(uniqueGeneralDialOffsets, generalDialReuseCounts, packedModel_.generalDialDescriptors, packedModel_.generalCachedDialCount);
  packDescriptors(uniqueGraphDialOffsets, graphDialReuseCounts, packedModel_.graphDialDescriptors, packedModel_.graphCachedDialCount);
  LogInfo << logPrefix_ << ": second pass completed in "
          << secondsSince(lastStageStart) << " s."
          << std::endl;
  buildTiming_.buildSecondPassSeconds = secondsSince(lastStageStart);
  lastStageStart = std::chrono::steady_clock::now();

  LogInfo << logPrefix_ << ": final flattening pass."
          << " This phase fills per-event offsets/counts and references to unique dials."
          << std::endl;
  packingLoopStart = std::chrono::steady_clock::now();
  lastPackingProgress = packingLoopStart;
  processedDialRefs = 0;
  for( std::size_t iEvent = 0 ; iEvent < model_.events.size() ; iEvent++ ){
    const auto& event = model_.events[iEvent];
    packedModel_.baseWeights[event.resultIndex] = float(event.weight.baseWeight);
    packedModel_.globalBins[event.resultIndex] = event.globalBinIndex;
    eventsPerBin[event.globalBinIndex]++;
    auto& eventRanges = packedModel_.eventDialRanges[event.resultIndex];
    eventRanges.normOffset = std::uint32_t(packedModel_.normDialOccurrences.size());
    eventRanges.compactOffset = std::uint32_t(packedModel_.compactDialIndices.size());
    eventRanges.uniformOffset = std::uint32_t(packedModel_.uniformDialIndices.size());
    eventRanges.monotonicOffset = std::uint32_t(packedModel_.monotonicDialIndices.size());
    eventRanges.generalOffset = std::uint32_t(packedModel_.generalDialIndices.size());
    eventRanges.graphOffset = std::uint32_t(packedModel_.graphDialIndices.size());
    for( std::size_t iDial = 0 ; iDial < event.weight.dialCount ; iDial++ ){
      processedDialRefs++;
      const auto& eventDial = model_.dials[model_.eventDialIndices[event.weight.firstDial + iDial]];
      if( eventDial.type == BackendDialType::Shift ){
        LogThrowIf(eventDial.payloadSize < 1, "Internal device packing error: Shift dial payload is empty.");
        packedModel_.baseWeights[event.resultIndex] *= float(model_.dialPayloads.at(eventDial.payloadOffset));
        continue;
      }
      if( eventDial.type == BackendDialType::Norm ){
        LogThrowIf(eventDial.inputCount != 1,
                   "Internal device packing error: Norm dial is missing its parameter input.");

        float minResponse = -std::numeric_limits<float>::infinity();
        float maxResponse = std::numeric_limits<float>::infinity();
        if( eventDial.hasMinResponse ){ minResponse = float(eventDial.minResponse); }
        if( eventDial.hasMaxResponse ){ maxResponse = float(eventDial.maxResponse); }

        DeviceNormDialOccurrence occurrence;
        occurrence.parameterIndex = std::uint32_t(model_.dialInputs.at(eventDial.firstInput).parameterIndex);
        occurrence.minResponse = minResponse;
        occurrence.maxResponse = maxResponse;
        packedModel_.normDialOccurrences.emplace_back(occurrence);
        continue;
      }
      auto packedDialIndexIt = packedDialIndexMap.find(eventDial.payloadOffset);
      LogThrowIf(packedDialIndexIt == packedDialIndexMap.end(), "Internal device packing error: missing unique dial index.");
      auto packedDialRef = packedDialIndexIt->second;
      switch( packedDialRef.type ){
        case kDeviceDialTypeCompactSpline: packedModel_.compactDialIndices.emplace_back(packedDialRef.localIndex); break;
        case kDeviceDialTypeUniformSpline: packedModel_.uniformDialIndices.emplace_back(packedDialRef.localIndex); break;
        case kDeviceDialTypeMonotonicSpline: packedModel_.monotonicDialIndices.emplace_back(packedDialRef.localIndex); break;
        case kDeviceDialTypeGeneralSpline: packedModel_.generalDialIndices.emplace_back(packedDialRef.localIndex); break;
        case kDeviceDialTypeGraph: packedModel_.graphDialIndices.emplace_back(packedDialRef.localIndex); break;
        default: LogThrow("Internal device packing error: unsupported event dial type during final flattening.");
      }
    }
    eventRanges.normCount = std::uint32_t(packedModel_.normDialOccurrences.size()) - eventRanges.normOffset;
    eventRanges.compactCount = std::uint32_t(packedModel_.compactDialIndices.size()) - eventRanges.compactOffset;
    eventRanges.uniformCount = std::uint32_t(packedModel_.uniformDialIndices.size()) - eventRanges.uniformOffset;
    eventRanges.monotonicCount = std::uint32_t(packedModel_.monotonicDialIndices.size()) - eventRanges.monotonicOffset;
    eventRanges.generalCount = std::uint32_t(packedModel_.generalDialIndices.size()) - eventRanges.generalOffset;
    eventRanges.graphCount = std::uint32_t(packedModel_.graphDialIndices.size()) - eventRanges.graphOffset;

    if( ((iEvent + 1) % kPackingProgressEventStep) == 0 or (iEvent + 1) == model_.events.size() ){
      auto now = std::chrono::steady_clock::now();
      LogInfo << logPrefix_ << ": final pass progress "
              << (iEvent + 1) << "/" << model_.events.size()
              << " events, "
              << processedDialRefs << "/" << model_.eventDialIndices.size()
              << " dial refs scanned, elapsed "
              << secondsSince(packingLoopStart) << " s"
              << " (+" << std::chrono::duration<double>(now - lastPackingProgress).count() << " s)"
              << "."
              << std::endl;
      lastPackingProgress = now;
    }
  }
  LogInfo << logPrefix_ << ": packed event/dial data in "
          << secondsSince(lastStageStart) << " s"
          << " [norm=" << normCount
          << ", compact=" << compactSplineCount
          << ", uniform=" << uniformSplineCount
          << ", monotonic=" << monotonicSplineCount
          << ", general=" << generalSplineCount
          << ", graph=" << graphCount
          << ", shift=" << shiftCount
          << ", unique dynamic packed="
          << (packedModel_.compactDialDescriptors.size() + packedModel_.uniformDialDescriptors.size()
              + packedModel_.monotonicDialDescriptors.size() + packedModel_.generalDialDescriptors.size()
              + packedModel_.graphDialDescriptors.size())
          << ", cached expensive dials=" << packedModel_.cachedDialCount
          << ", event dial occurrences=" << packedModel_.totalDynamicDialOccurrences
          << ", spline scalars=" << packedModel_.splineData.size()
          << "]."
          << std::endl;
  buildTiming_.buildFinalFlattenSeconds = secondsSince(lastStageStart);
  lastStageStart = std::chrono::steady_clock::now();

  packedModel_.binEventOffsets.assign(model_.totalBins + 1, 0);
  for( int iBin = 0 ; iBin < model_.totalBins ; iBin++ ){
    packedModel_.binEventOffsets[iBin + 1] = packedModel_.binEventOffsets[iBin] + eventsPerBin[iBin];
  }
  std::vector<std::uint32_t> binEventFill = packedModel_.binEventOffsets;
  packedModel_.binEventIndices.resize(model_.events.size());
  for( const auto& event : model_.events ){
    auto& fillIndex = binEventFill[event.globalBinIndex];
    packedModel_.binEventIndices[fillIndex++] = std::uint32_t(event.resultIndex);
  }

  for( auto count : eventsPerBin ){
    packedModel_.maxEventsPerBin = std::max(packedModel_.maxEventsPerBin, count);
  }
  packedModel_.maxHistogramChunksPerBin = std::max<std::uint32_t>(
      1,
      (packedModel_.maxEventsPerBin + kDeviceHistogramChunkSize - 1) / kDeviceHistogramChunkSize
  );
  LogInfo << logPrefix_ << ": built histogram index tables in "
          << secondsSince(lastStageStart) << " s"
          << " [max events/bin=" << packedModel_.maxEventsPerBin
          << ", chunks/bin=" << packedModel_.maxHistogramChunksPerBin
          << "]."
          << std::endl;
  buildTiming_.buildHistogramIndexSeconds = secondsSince(lastStageStart);

  packedModel_.compactDialDescriptorCount = std::uint32_t(packedModel_.compactDialDescriptors.size());
  packedModel_.uniformDialDescriptorCount = std::uint32_t(packedModel_.uniformDialDescriptors.size());
  packedModel_.monotonicDialDescriptorCount = std::uint32_t(packedModel_.monotonicDialDescriptors.size());
  packedModel_.generalDialDescriptorCount = std::uint32_t(packedModel_.generalDialDescriptors.size());
  packedModel_.graphDialDescriptorCount = std::uint32_t(packedModel_.graphDialDescriptors.size());

  if( packedModel_.normDialOccurrences.empty() ){ packedModel_.normDialOccurrences.emplace_back(DeviceNormDialOccurrence{}); }
  if( packedModel_.compactDialIndices.empty() ){ packedModel_.compactDialIndices.emplace_back(0); }
  if( packedModel_.uniformDialIndices.empty() ){ packedModel_.uniformDialIndices.emplace_back(0); }
  if( packedModel_.monotonicDialIndices.empty() ){ packedModel_.monotonicDialIndices.emplace_back(0); }
  if( packedModel_.generalDialIndices.empty() ){ packedModel_.generalDialIndices.emplace_back(0); }
  if( packedModel_.graphDialIndices.empty() ){ packedModel_.graphDialIndices.emplace_back(0); }
  if( packedModel_.compactDialDescriptors.empty() ){ packedModel_.compactDialDescriptors.emplace_back(DeviceSplineDialDescriptor{}); }
  if( packedModel_.uniformDialDescriptors.empty() ){ packedModel_.uniformDialDescriptors.emplace_back(DeviceSplineDialDescriptor{}); }
  if( packedModel_.monotonicDialDescriptors.empty() ){ packedModel_.monotonicDialDescriptors.emplace_back(DeviceSplineDialDescriptor{}); }
  if( packedModel_.generalDialDescriptors.empty() ){ packedModel_.generalDialDescriptors.emplace_back(DeviceSplineDialDescriptor{}); }
  if( packedModel_.graphDialDescriptors.empty() ){ packedModel_.graphDialDescriptors.emplace_back(DeviceSplineDialDescriptor{}); }
  if( packedModel_.splineData.empty() ){ packedModel_.splineData.emplace_back(0); }

  buildTiming_.uniqueDialCount = packedModel_.uniqueDialCount();
  buildTiming_.cachedDialCount = packedModel_.cachedDialCount;
  buildTiming_.eventDialIndexCount = packedModel_.totalDynamicDialOccurrences;
  buildTiming_.splineScalarCount = packedModel_.splineData.size();
  LogInfo << logPrefix_ << ": packed device model completed in "
          << secondsSince(buildStart) << " s."
          << std::endl;
  return true;
}
