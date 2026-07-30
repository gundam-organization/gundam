#include "MpsBackendInternal.h"

bool Backends::MpsBackendImpl::buildDeviceModel() {
  releaseDeviceBuffers();
  isDeviceModelSupported = false;
  deviceModelFallbackReason.clear();
  auto buildStart = std::chrono::steady_clock::now();
  auto lastStageStart = buildStart;

  auto fail = [this](std::string reason_) {
    deviceModelFallbackReason = std::move(reason_);
    return false;
  };

  if( not isAvailable ){ return fail("Metal is not available."); }
  if( model.events.empty() ){
    return fail("the backend model has no events.");
  }
  if( model.totalBins <= 0 ){
    return fail("the backend model has no histogram bins.");
  }

  LogInfo << "MPS backend: building device model for "
          << model.events.size() << " events, "
          << model.eventDials.size() << " event dials, "
          << model.parameterCount << " parameters and "
          << model.totalBins << " histogram bins."
          << std::endl;

  LogInfo << "MPS backend: compatibility scan done in "
          << secondsSince(lastStageStart) << " s."
          << std::endl;
  buildTiming.buildCompatibilityScanSeconds = secondsSince(lastStageStart);
  lastStageStart = std::chrono::steady_clock::now();
  LogInfo << "MPS backend: built parameter lookup table in "
          << secondsSince(lastStageStart) << " s."
          << std::endl;
  buildTiming.buildParameterLookupSeconds = secondsSince(lastStageStart);
  lastStageStart = std::chrono::steady_clock::now();

  std::vector<float> baseWeights(model.events.size());
  std::vector<MpsEventDialRanges> eventDialRanges(model.events.size());
  std::vector<MpsNormDialOccurrence> normDialOccurrences{};
  std::vector<uint32_t> compactDialIndices{};
  std::vector<uint32_t> uniformDialIndices{};
  std::vector<uint32_t> monotonicDialIndices{};
  std::vector<uint32_t> generalDialIndices{};
  std::vector<uint32_t> graphDialIndices{};
  std::vector<MpsSplineDialDescriptor> compactDialDescriptors{};
  std::vector<MpsSplineDialDescriptor> uniformDialDescriptors{};
  std::vector<MpsSplineDialDescriptor> monotonicDialDescriptors{};
  std::vector<MpsSplineDialDescriptor> generalDialDescriptors{};
  std::vector<MpsSplineDialDescriptor> graphDialDescriptors{};
  std::size_t totalDynamicDialOccurrences{0};
  std::vector<int> globalBins(model.events.size());
  std::vector<uint32_t> eventsPerBin(model.totalBins, 0);
  std::vector<float> splineData{};
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
  std::vector<uint32_t> compactDialReuseCounts{};
  std::vector<uint32_t> uniformDialReuseCounts{};
  std::vector<uint32_t> monotonicDialReuseCounts{};
  std::vector<uint32_t> generalDialReuseCounts{};
  std::vector<uint32_t> graphDialReuseCounts{};
  uniqueCompactDialOffsets.reserve(model.eventDials.size());
  uniqueUniformDialOffsets.reserve(model.eventDials.size());
  uniqueMonotonicDialOffsets.reserve(model.eventDials.size());
  uniqueGeneralDialOffsets.reserve(model.eventDials.size());
  uniqueGraphDialOffsets.reserve(model.eventDials.size());
  compactDialReuseCounts.reserve(model.eventDials.size());
  uniformDialReuseCounts.reserve(model.eventDials.size());
  monotonicDialReuseCounts.reserve(model.eventDials.size());
  generalDialReuseCounts.reserve(model.eventDials.size());
  graphDialReuseCounts.reserve(model.eventDials.size());
  std::unordered_map<std::size_t, MpsPackedDialRef> packedDialIndexMap{};
  packedDialIndexMap.reserve(model.eventDials.size());
  std::unordered_map<std::size_t, const BackendDialView*> packedDialDescriptorMap{};
  packedDialDescriptorMap.reserve(model.eventDials.size());

  LogInfo << "MPS backend: first packing pass."
          << " This phase inventories unique shared dials and precomputes payload sizes."
          << std::endl;

  constexpr std::size_t kPackingProgressEventStep = 10000;
  auto packingLoopStart = std::chrono::steady_clock::now();
  auto lastPackingProgress = packingLoopStart;
  std::size_t processedDialRefs{0};
  for( std::size_t iEvent = 0 ; iEvent < model.events.size() ; iEvent++ ){
    const auto& event = model.events[iEvent];
    if( event.globalBinIndex < 0 or event.globalBinIndex >= model.totalBins ){
      return fail("at least one event has an invalid global bin index.");
    }
    for( std::size_t iDial = 0 ; iDial < event.dialCount ; iDial++ ){
      processedDialRefs++;
      const auto& eventDial = model.eventDials[event.firstDial + iDial];
      if( eventDial.type == BackendDialType::Shift ){
        shiftCount++;
        continue;
      }

      if( eventDial.inputCount != 1 ){
        return fail("at least one backend dial is not MPS-compatible because it does not have exactly one input parameter.");
      }

      if( eventDial.type == BackendDialType::Norm ){
        normCount++;
        totalDynamicDialOccurrences++;
        continue;
      }

      auto packedDialIndexIt = packedDialIndexMap.find(eventDial.payloadOffset);
      if( packedDialIndexIt != packedDialIndexMap.end() ){
        auto packedDialRef = packedDialIndexIt->second;
        switch( packedDialRef.type ){
          case kMpsDialTypeCompactSpline: compactDialReuseCounts[packedDialRef.localIndex]++; break;
          case kMpsDialTypeUniformSpline: uniformDialReuseCounts[packedDialRef.localIndex]++; break;
          case kMpsDialTypeMonotonicSpline: monotonicDialReuseCounts[packedDialRef.localIndex]++; break;
          case kMpsDialTypeGeneralSpline: generalDialReuseCounts[packedDialRef.localIndex]++; break;
          case kMpsDialTypeGraph: graphDialReuseCounts[packedDialRef.localIndex]++; break;
          default: LogThrow("Internal MPS packing error: unexpected dial type in reuse table.");
        }
        totalDynamicDialOccurrences++;
        continue;
      }

      MpsPackedDialRef packedDialRef;
      totalDynamicDialOccurrences++;
      if( eventDial.type == BackendDialType::CompactSpline ){
        if( eventDial.payloadSize < 6 ){
          return fail("CompactSpline dial data is too small for MPS evaluation.");
        }
        uniqueSplineScalarCount += eventDial.payloadSize;
        compactSplineCount++;
        packedDialRef.type = kMpsDialTypeCompactSpline;
        packedDialRef.localIndex = uint32_t(uniqueCompactDialOffsets.size());
        uniqueCompactDialOffsets.emplace_back(eventDial.payloadOffset);
        compactDialReuseCounts.emplace_back(1);
      }
      else if( eventDial.type == BackendDialType::UniformSpline ){
        if( eventDial.payloadSize < 8 ){
          return fail("UniformSpline dial data is too small for MPS evaluation.");
        }
        uniqueSplineScalarCount += eventDial.payloadSize;
        uniformSplineCount++;
        packedDialRef.type = kMpsDialTypeUniformSpline;
        packedDialRef.localIndex = uint32_t(uniqueUniformDialOffsets.size());
        uniqueUniformDialOffsets.emplace_back(eventDial.payloadOffset);
        uniformDialReuseCounts.emplace_back(1);
      }
      else if( eventDial.type == BackendDialType::MonotonicSpline ){
        if( eventDial.payloadSize < 5 ){
          return fail("MonotonicSpline dial data is too small for MPS evaluation.");
        }
        uniqueSplineScalarCount += eventDial.payloadSize;
        monotonicSplineCount++;
        packedDialRef.type = kMpsDialTypeMonotonicSpline;
        packedDialRef.localIndex = uint32_t(uniqueMonotonicDialOffsets.size());
        uniqueMonotonicDialOffsets.emplace_back(eventDial.payloadOffset);
        monotonicDialReuseCounts.emplace_back(1);
      }
      else if( eventDial.type == BackendDialType::GeneralSpline ){
        if( eventDial.payloadSize < 11 ){
          return fail("GeneralSpline dial data is too small for MPS evaluation.");
        }
        uniqueSplineScalarCount += eventDial.payloadSize;
        generalSplineCount++;
        packedDialRef.type = kMpsDialTypeGeneralSpline;
        packedDialRef.localIndex = uint32_t(uniqueGeneralDialOffsets.size());
        uniqueGeneralDialOffsets.emplace_back(eventDial.payloadOffset);
        generalDialReuseCounts.emplace_back(1);
      }
      else if( eventDial.type == BackendDialType::Graph ){
        if( eventDial.payloadSize < 2 ){
          return fail("Graph dial data is too small for MPS evaluation.");
        }
        uniqueSplineScalarCount += eventDial.payloadSize;
        graphCount++;
        packedDialRef.type = kMpsDialTypeGraph;
        packedDialRef.localIndex = uint32_t(uniqueGraphDialOffsets.size());
        uniqueGraphDialOffsets.emplace_back(eventDial.payloadOffset);
        graphDialReuseCounts.emplace_back(1);
      }
      else{
        return fail("unexpected backend dial type has no MPS device encoder.");
      }
      packedDialIndexMap.emplace(eventDial.payloadOffset, packedDialRef);
      packedDialDescriptorMap.emplace(eventDial.payloadOffset, &eventDial);
    }

    if( ((iEvent + 1) % kPackingProgressEventStep) == 0 or (iEvent + 1) == model.events.size() ){
      auto now = std::chrono::steady_clock::now();
      LogInfo << "MPS backend: first pass progress "
              << (iEvent + 1) << "/" << model.events.size()
              << " events, "
              << processedDialRefs << "/" << model.eventDials.size()
              << " dial refs scanned, "
              << packedDialIndexMap.size() << " unique dynamic dials found, elapsed "
              << secondsSince(packingLoopStart) << " s"
              << " (+" << std::chrono::duration<double>(now - lastPackingProgress).count() << " s)"
              << "."
              << std::endl;
      lastPackingProgress = now;
    }
  }
  LogInfo << "MPS backend: first pass completed in "
          << secondsSince(lastStageStart) << " s"
          << " [unique dynamic dials=" << packedDialIndexMap.size()
          << ", unique spline scalars=" << uniqueSplineScalarCount
          << "]."
          << std::endl;
  buildTiming.buildFirstPassSeconds = secondsSince(lastStageStart);
  lastStageStart = std::chrono::steady_clock::now();

  compactDialDescriptors.reserve(uniqueCompactDialOffsets.size());
  uniformDialDescriptors.reserve(uniqueUniformDialOffsets.size());
  monotonicDialDescriptors.reserve(uniqueMonotonicDialOffsets.size());
  generalDialDescriptors.reserve(uniqueGeneralDialOffsets.size());
  graphDialDescriptors.reserve(uniqueGraphDialOffsets.size());
  normDialOccurrences.reserve(normCount);
  compactDialIndices.reserve(compactSplineCount);
  uniformDialIndices.reserve(uniformSplineCount);
  monotonicDialIndices.reserve(monotonicSplineCount);
  generalDialIndices.reserve(generalSplineCount);
  graphDialIndices.reserve(graphCount);
  splineData.reserve(uniqueSplineScalarCount);

  LogInfo << "MPS backend: second packing pass."
          << " This phase materializes unique dial descriptors and payloads."
          << std::endl;
  cachedDialCount = 0;
  auto fillMinMax = [](const BackendDialView& dialRef_, float& minResponse_, float& maxResponse_) {
    minResponse_ = -std::numeric_limits<float>::infinity();
    maxResponse_ = std::numeric_limits<float>::infinity();
    if( dialRef_.hasMinResponse ){ minResponse_ = float(dialRef_.minResponse); }
    if( dialRef_.hasMaxResponse ){ maxResponse_ = float(dialRef_.maxResponse); }
  };
  auto packDescriptors = [&](const std::vector<std::size_t>& dialOffsets_,
                             const std::vector<uint32_t>& reuseCounts_,
                             std::vector<MpsSplineDialDescriptor>& descriptors_,
                             uint32_t& cachedCount_) {
    for( std::size_t iUniqueDial = 0 ; iUniqueDial < dialOffsets_.size() ; iUniqueDial++ ){
      auto eventDialIt = packedDialDescriptorMap.find(dialOffsets_[iUniqueDial]);
      LogThrowIf(eventDialIt == packedDialDescriptorMap.end(),
                 "Internal MPS packing error: could not resolve unique backend dial descriptor.");
      const auto& eventDial = *eventDialIt->second;

      MpsSplineDialDescriptor descriptor;
      descriptor.parameterIndex = uint32_t(model.dialInputs.at(eventDial.firstInput).parameterIndex);
      descriptor.splineOffset = uint32_t(splineData.size());
      descriptor.splineSize = uint32_t(eventDial.payloadSize);
      descriptor.flags = eventDial.allowExtrapolation ? kMpsDialFlagAllowExtrapolation : 0u;
      fillMinMax(eventDial, descriptor.minResponse, descriptor.maxResponse);

      if( reuseCounts_[iUniqueDial] >= kMpsCachedDialReuseThreshold ){
        descriptor.flags |= kMpsDialFlagCached;
        cachedCount_++;
        cachedDialCount++;
      }

      for( std::size_t iPayload = 0 ; iPayload < eventDial.payloadSize ; iPayload++ ){
        splineData.emplace_back(float(model.dialPayloads.at(eventDial.payloadOffset + iPayload)));
      }
      descriptors_.emplace_back(descriptor);
    }
  };

  compactCachedDialCount = 0;
  uniformCachedDialCount = 0;
  monotonicCachedDialCount = 0;
  generalCachedDialCount = 0;
  graphCachedDialCount = 0;
  packDescriptors(uniqueCompactDialOffsets, compactDialReuseCounts, compactDialDescriptors, compactCachedDialCount);
  packDescriptors(uniqueUniformDialOffsets, uniformDialReuseCounts, uniformDialDescriptors, uniformCachedDialCount);
  packDescriptors(uniqueMonotonicDialOffsets, monotonicDialReuseCounts, monotonicDialDescriptors, monotonicCachedDialCount);
  packDescriptors(uniqueGeneralDialOffsets, generalDialReuseCounts, generalDialDescriptors, generalCachedDialCount);
  packDescriptors(uniqueGraphDialOffsets, graphDialReuseCounts, graphDialDescriptors, graphCachedDialCount);
  LogInfo << "MPS backend: second pass completed in "
          << secondsSince(lastStageStart) << " s."
          << std::endl;
  buildTiming.buildSecondPassSeconds = secondsSince(lastStageStart);
  lastStageStart = std::chrono::steady_clock::now();

  LogInfo << "MPS backend: final flattening pass."
          << " This phase fills per-event offsets/counts and references to unique dials."
          << std::endl;
  packingLoopStart = std::chrono::steady_clock::now();
  lastPackingProgress = packingLoopStart;
  processedDialRefs = 0;
  for( std::size_t iEvent = 0 ; iEvent < model.events.size() ; iEvent++ ){
    const auto& event = model.events[iEvent];
    baseWeights[event.resultIndex] = float(event.baseWeight);
    globalBins[event.resultIndex] = event.globalBinIndex;
    eventsPerBin[event.globalBinIndex]++;
    auto& eventRanges = eventDialRanges[event.resultIndex];
    eventRanges.normOffset = uint32_t(normDialOccurrences.size());
    eventRanges.compactOffset = uint32_t(compactDialIndices.size());
    eventRanges.uniformOffset = uint32_t(uniformDialIndices.size());
    eventRanges.monotonicOffset = uint32_t(monotonicDialIndices.size());
    eventRanges.generalOffset = uint32_t(generalDialIndices.size());
    eventRanges.graphOffset = uint32_t(graphDialIndices.size());
    for( std::size_t iDial = 0 ; iDial < event.dialCount ; iDial++ ){
      processedDialRefs++;
      const auto& eventDial = model.eventDials[event.firstDial + iDial];
      if( eventDial.type == BackendDialType::Shift ){
        LogThrowIf(eventDial.payloadSize < 1, "Internal MPS packing error: Shift dial payload is empty.");
        baseWeights[event.resultIndex] *= float(model.dialPayloads.at(eventDial.payloadOffset));
        continue;
      }
      if( eventDial.type == BackendDialType::Norm ){
        LogThrowIf(eventDial.inputCount != 1,
                   "Internal MPS packing error: Norm dial is missing its parameter input.");

        float minResponse = -std::numeric_limits<float>::infinity();
        float maxResponse = std::numeric_limits<float>::infinity();
        if( eventDial.hasMinResponse ){ minResponse = float(eventDial.minResponse); }
        if( eventDial.hasMaxResponse ){ maxResponse = float(eventDial.maxResponse); }

        MpsNormDialOccurrence occurrence;
        occurrence.parameterIndex = uint32_t(model.dialInputs.at(eventDial.firstInput).parameterIndex);
        occurrence.minResponse = minResponse;
        occurrence.maxResponse = maxResponse;
        normDialOccurrences.emplace_back(occurrence);
        continue;
      }
      auto packedDialIndexIt = packedDialIndexMap.find(eventDial.payloadOffset);
      LogThrowIf(packedDialIndexIt == packedDialIndexMap.end(), "Internal MPS packing error: missing unique dial index.");
      auto packedDialRef = packedDialIndexIt->second;
      switch( packedDialRef.type ){
        case kMpsDialTypeCompactSpline: compactDialIndices.emplace_back(packedDialRef.localIndex); break;
        case kMpsDialTypeUniformSpline: uniformDialIndices.emplace_back(packedDialRef.localIndex); break;
        case kMpsDialTypeMonotonicSpline: monotonicDialIndices.emplace_back(packedDialRef.localIndex); break;
        case kMpsDialTypeGeneralSpline: generalDialIndices.emplace_back(packedDialRef.localIndex); break;
        case kMpsDialTypeGraph: graphDialIndices.emplace_back(packedDialRef.localIndex); break;
        default: LogThrow("Internal MPS packing error: unsupported event dial type during final flattening.");
      }
    }
    eventRanges.normCount = uint32_t(normDialOccurrences.size()) - eventRanges.normOffset;
    eventRanges.compactCount = uint32_t(compactDialIndices.size()) - eventRanges.compactOffset;
    eventRanges.uniformCount = uint32_t(uniformDialIndices.size()) - eventRanges.uniformOffset;
    eventRanges.monotonicCount = uint32_t(monotonicDialIndices.size()) - eventRanges.monotonicOffset;
    eventRanges.generalCount = uint32_t(generalDialIndices.size()) - eventRanges.generalOffset;
    eventRanges.graphCount = uint32_t(graphDialIndices.size()) - eventRanges.graphOffset;

    if( ((iEvent + 1) % kPackingProgressEventStep) == 0 or (iEvent + 1) == model.events.size() ){
      auto now = std::chrono::steady_clock::now();
      LogInfo << "MPS backend: final pass progress "
              << (iEvent + 1) << "/" << model.events.size()
              << " events, "
              << processedDialRefs << "/" << model.eventDials.size()
              << " dial refs scanned, elapsed "
              << secondsSince(packingLoopStart) << " s"
              << " (+" << std::chrono::duration<double>(now - lastPackingProgress).count() << " s)"
              << "."
              << std::endl;
      lastPackingProgress = now;
    }
  }
  LogInfo << "MPS backend: packed event/dial data in "
          << secondsSince(lastStageStart) << " s"
          << " [norm=" << normCount
          << ", compact=" << compactSplineCount
          << ", uniform=" << uniformSplineCount
          << ", monotonic=" << monotonicSplineCount
          << ", general=" << generalSplineCount
          << ", graph=" << graphCount
          << ", shift=" << shiftCount
          << ", unique dynamic packed="
          << (compactDialDescriptors.size() + uniformDialDescriptors.size()
              + monotonicDialDescriptors.size() + generalDialDescriptors.size()
              + graphDialDescriptors.size())
          << ", cached expensive dials=" << cachedDialCount
          << ", event dial occurrences=" << totalDynamicDialOccurrences
          << ", spline scalars=" << splineData.size()
          << "]."
          << std::endl;
  buildTiming.buildFinalFlattenSeconds = secondsSince(lastStageStart);
  lastStageStart = std::chrono::steady_clock::now();

  std::vector<uint32_t> binEventOffsets(model.totalBins + 1, 0);
  for( int iBin = 0 ; iBin < model.totalBins ; iBin++ ){
    binEventOffsets[iBin + 1] = binEventOffsets[iBin] + eventsPerBin[iBin];
  }
  std::vector<uint32_t> binEventFill = binEventOffsets;
  std::vector<uint32_t> binEventIndices(model.events.size());
  for( const auto& event : model.events ){
    auto& fillIndex = binEventFill[event.globalBinIndex];
    binEventIndices[fillIndex++] = uint32_t(event.resultIndex);
  }

  uint32_t maxEventsPerBin = 0;
  for( auto count : eventsPerBin ){
    maxEventsPerBin = std::max(maxEventsPerBin, count);
  }
  maxHistogramChunksPerBin = std::max<uint32_t>(
      1,
      (maxEventsPerBin + histogramChunkSize - 1) / histogramChunkSize
  );
  LogInfo << "MPS backend: built histogram index tables in "
          << secondsSince(lastStageStart) << " s"
          << " [max events/bin=" << maxEventsPerBin
          << ", chunks/bin=" << maxHistogramChunksPerBin
          << "]."
          << std::endl;
  buildTiming.buildHistogramIndexSeconds = secondsSince(lastStageStart);
  lastStageStart = std::chrono::steady_clock::now();

  compactDialDescriptorCount = uint32_t(compactDialDescriptors.size());
  uniformDialDescriptorCount = uint32_t(uniformDialDescriptors.size());
  monotonicDialDescriptorCount = uint32_t(monotonicDialDescriptors.size());
  generalDialDescriptorCount = uint32_t(generalDialDescriptors.size());
  graphDialDescriptorCount = uint32_t(graphDialDescriptors.size());
  if( normDialOccurrences.empty() ){ normDialOccurrences.emplace_back(MpsNormDialOccurrence{}); }
  if( compactDialIndices.empty() ){ compactDialIndices.emplace_back(0); }
  if( uniformDialIndices.empty() ){ uniformDialIndices.emplace_back(0); }
  if( monotonicDialIndices.empty() ){ monotonicDialIndices.emplace_back(0); }
  if( generalDialIndices.empty() ){ generalDialIndices.emplace_back(0); }
  if( graphDialIndices.empty() ){ graphDialIndices.emplace_back(0); }
  if( compactDialDescriptors.empty() ){ compactDialDescriptors.emplace_back(MpsSplineDialDescriptor{}); }
  if( uniformDialDescriptors.empty() ){ uniformDialDescriptors.emplace_back(MpsSplineDialDescriptor{}); }
  if( monotonicDialDescriptors.empty() ){ monotonicDialDescriptors.emplace_back(MpsSplineDialDescriptor{}); }
  if( generalDialDescriptors.empty() ){ generalDialDescriptors.emplace_back(MpsSplineDialDescriptor{}); }
  if( graphDialDescriptors.empty() ){ graphDialDescriptors.emplace_back(MpsSplineDialDescriptor{}); }
  if( splineData.empty() ){ splineData.emplace_back(0); }

  uint32_t nEvents = uint32_t(model.events.size());
  uint32_t totalBins = uint32_t(model.totalBins);
  uint32_t chunkSize = histogramChunkSize;
  uint32_t totalPartials = totalBins * maxHistogramChunksPerBin;

  LogInfo << "MPS backend: allocating/uploading Metal buffers."
          << " Event weights bytes=" << model.events.size() * sizeof(float)
          << ", histogram bytes=" << std::size_t(model.totalBins) * sizeof(float)
          << ", partial histogram bytes=" << std::size_t(totalPartials) * sizeof(float)
          << ", unique packed dial count="
          << (compactDialDescriptorCount + uniformDialDescriptorCount
              + monotonicDialDescriptorCount + generalDialDescriptorCount
              + graphDialDescriptorCount)
          << ", cached dial count=" << cachedDialCount
          << ", event dial occurrence count=" << totalDynamicDialOccurrences
          << "."
          << std::endl;

  baseWeightsBuffer = makePrivateBuffer(device, commandQueue, baseWeights);
  eventDialRangesBuffer = makePrivateBuffer(device, commandQueue, eventDialRanges);
  normDialOccurrencesBuffer = makePrivateBuffer(device, commandQueue, normDialOccurrences);
  compactDialIndicesBuffer = makePrivateBuffer(device, commandQueue, compactDialIndices);
  uniformDialIndicesBuffer = makePrivateBuffer(device, commandQueue, uniformDialIndices);
  monotonicDialIndicesBuffer = makePrivateBuffer(device, commandQueue, monotonicDialIndices);
  generalDialIndicesBuffer = makePrivateBuffer(device, commandQueue, generalDialIndices);
  graphDialIndicesBuffer = makePrivateBuffer(device, commandQueue, graphDialIndices);
  compactDialDescriptorsBuffer = makePrivateBuffer(device, commandQueue, compactDialDescriptors);
  uniformDialDescriptorsBuffer = makePrivateBuffer(device, commandQueue, uniformDialDescriptors);
  monotonicDialDescriptorsBuffer = makePrivateBuffer(device, commandQueue, monotonicDialDescriptors);
  generalDialDescriptorsBuffer = makePrivateBuffer(device, commandQueue, generalDialDescriptors);
  graphDialDescriptorsBuffer = makePrivateBuffer(device, commandQueue, graphDialDescriptors);
  globalBinsBuffer = makePrivateBuffer(device, commandQueue, globalBins);
  binEventOffsetsBuffer = makePrivateBuffer(device, commandQueue, binEventOffsets);
  binEventIndicesBuffer = makePrivateBuffer(device, commandQueue, binEventIndices);
  splineDataBuffer = makePrivateBuffer(device, commandQueue, splineData);
  compactCachedResponsesBuffer = makePrivateEmptyBuffer(device, std::size_t(compactDialDescriptors.size()) * sizeof(float));
  uniformCachedResponsesBuffer = makePrivateEmptyBuffer(device, std::size_t(uniformDialDescriptors.size()) * sizeof(float));
  monotonicCachedResponsesBuffer = makePrivateEmptyBuffer(device, std::size_t(monotonicDialDescriptors.size()) * sizeof(float));
  generalCachedResponsesBuffer = makePrivateEmptyBuffer(device, std::size_t(generalDialDescriptors.size()) * sizeof(float));
  graphCachedResponsesBuffer = makePrivateEmptyBuffer(device, std::size_t(graphDialDescriptors.size()) * sizeof(float));
  eventWeightsBuffer = makePrivateEmptyBuffer(device, model.events.size() * sizeof(float));
  eventWeightsReadbackBuffer = makeSharedEmptyBuffer(device, model.events.size() * sizeof(float));
  parametersBuffer = makeSharedEmptyBuffer(device, std::max<std::size_t>(1, model.parameterCount) * sizeof(float));
  partialHistSumsBuffer = makePrivateEmptyBuffer(device, std::size_t(totalPartials) * sizeof(float));
  partialHistSumSquaresBuffer = makePrivateEmptyBuffer(device, std::size_t(totalPartials) * sizeof(float));
  histSumsBuffer = makePrivateEmptyBuffer(device, std::size_t(model.totalBins) * sizeof(float));
  histSumSquaresBuffer = makePrivateEmptyBuffer(device, std::size_t(model.totalBins) * sizeof(float));
  histSumsReadbackBuffer = makeSharedEmptyBuffer(device, std::size_t(model.totalBins) * sizeof(float));
  histSumSquaresReadbackBuffer = makeSharedEmptyBuffer(device, std::size_t(model.totalBins) * sizeof(float));
  nEventsBuffer = [device newBufferWithBytes:&nEvents length:sizeof(nEvents) options:MTLResourceStorageModeShared];
  totalBinsBuffer = [device newBufferWithBytes:&totalBins length:sizeof(totalBins) options:MTLResourceStorageModeShared];
  maxHistogramChunksPerBinBuffer = [device newBufferWithBytes:&maxHistogramChunksPerBin
                                                       length:sizeof(maxHistogramChunksPerBin)
                                                      options:MTLResourceStorageModeShared];
  histogramChunkSizeBuffer = [device newBufferWithBytes:&chunkSize
                                                 length:sizeof(chunkSize)
                                                options:MTLResourceStorageModeShared];

  if( baseWeightsBuffer == nil or eventDialRangesBuffer == nil or normDialOccurrencesBuffer == nil
      or compactDialIndicesBuffer == nil or uniformDialIndicesBuffer == nil
      or monotonicDialIndicesBuffer == nil or generalDialIndicesBuffer == nil
      or graphDialIndicesBuffer == nil
      or compactDialDescriptorsBuffer == nil or uniformDialDescriptorsBuffer == nil
      or monotonicDialDescriptorsBuffer == nil or generalDialDescriptorsBuffer == nil
      or graphDialDescriptorsBuffer == nil
      or compactCachedResponsesBuffer == nil or uniformCachedResponsesBuffer == nil
      or monotonicCachedResponsesBuffer == nil or generalCachedResponsesBuffer == nil
      or graphCachedResponsesBuffer == nil
      or globalBinsBuffer == nil or binEventOffsetsBuffer == nil or binEventIndicesBuffer == nil
      or splineDataBuffer == nil
      or eventWeightsBuffer == nil or parametersBuffer == nil
      or partialHistSumsBuffer == nil or partialHistSumSquaresBuffer == nil
      or histSumsBuffer == nil or histSumSquaresBuffer == nil or eventWeightsReadbackBuffer == nil
      or histSumsReadbackBuffer == nil or histSumSquaresReadbackBuffer == nil or nEventsBuffer == nil
      or totalBinsBuffer == nil or maxHistogramChunksPerBinBuffer == nil
      or histogramChunkSizeBuffer == nil ){
    releaseDeviceBuffers();
    return fail("Metal buffer allocation failed while building the MPS backend model.");
  }
  LogInfo << "MPS backend: Metal buffers ready in "
          << secondsSince(lastStageStart) << " s."
          << std::endl;
  buildTiming.buildBufferUploadSeconds = secondsSince(lastStageStart);

  parameterValuesScratch.resize(model.parameterCount);
  isDeviceModelSupported = true;
  buildTiming.uniqueDialCount = compactDialDescriptorCount + uniformDialDescriptorCount
                                + monotonicDialDescriptorCount + generalDialDescriptorCount
                                + graphDialDescriptorCount;
  buildTiming.cachedDialCount = cachedDialCount;
  buildTiming.eventDialIndexCount = totalDynamicDialOccurrences;
  buildTiming.splineScalarCount = splineData.size();
  LogInfo << "MPS backend: device model build completed in "
          << secondsSince(buildStart) << " s."
          << std::endl;
  return true;
}
