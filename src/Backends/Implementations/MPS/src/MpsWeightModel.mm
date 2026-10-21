#include "MpsBackendInternal.h"
#include "DeviceWeightModel.h"
#include "MpsSplinePayloadLayout.h"

#include <array>

bool Backends::MpsBackendImpl::buildWeightModel() {
  releaseDeviceBuffers();
  isWeightModelSupported = false;
  deviceModelFallbackReason.clear();
  if( not isAvailable ){ deviceModelFallbackReason = "Metal is not available."; return false; }
  DeviceWeightModel packed;
  if( not packDeviceWeightModel(model, enableCachedDialResponses, packed, buildTiming,
                                  deviceModelFallbackReason, "MPS backend") ){ return false; }
  if( enableDiagnosticDescriptorComparison ){ useOrderedSplinePayloads = false; }
  std::unique_ptr<MpsSplinePayloadLayout> ordered;
  if( useOrderedSplinePayloads or enableDiagnosticPayloadComparison ){
    ordered = std::make_unique<MpsSplinePayloadLayout>(packed);
    if( not enableDiagnosticPayloadComparison ){
      ordered->apply(packed);
      ordered.reset();
    }
  }
  auto lastStageStart = std::chrono::steady_clock::now();
  auto buildStart = lastStageStart;
  auto fail = [this](std::string reason_){ deviceModelFallbackReason = std::move(reason_); return false; };
  auto& baseWeights = packed.baseWeights;
  auto& eventDialRanges = packed.eventDialRanges;
  auto& normDialOccurrences = packed.normDialOccurrences;
  auto& externalDialOccurrences = packed.externalDialOccurrences;
  auto& compactDialIndices = packed.compactDialIndices;
  auto& uniformDialIndices = packed.uniformDialIndices;
  auto& monotonicDialIndices = packed.monotonicDialIndices;
  auto& generalDialIndices = packed.generalDialIndices;
  auto& graphDialIndices = packed.graphDialIndices;
  auto& compactDialDescriptors = packed.compactDialDescriptors;
  auto& uniformDialDescriptors = packed.uniformDialDescriptors;
  auto& monotonicDialDescriptors = packed.monotonicDialDescriptors;
  auto& generalDialDescriptors = packed.generalDialDescriptors;
  auto& graphDialDescriptors = packed.graphDialDescriptors;
  auto& globalBins = packed.globalBins;
  auto& binEventOffsets = packed.binEventOffsets;
  auto& binEventIndices = packed.binEventIndices;
  auto& splineData = packed.splineData;
  cachedDialCount = packed.cachedDialCount;
  compactCachedDialCount = packed.compactCachedDialCount;
  uniformCachedDialCount = packed.uniformCachedDialCount;
  monotonicCachedDialCount = packed.monotonicCachedDialCount;
  generalCachedDialCount = packed.generalCachedDialCount;
  graphCachedDialCount = packed.graphCachedDialCount;
  compactDialDescriptorCount = packed.compactDialDescriptorCount;
  uniformDialDescriptorCount = packed.uniformDialDescriptorCount;
  monotonicDialDescriptorCount = packed.monotonicDialDescriptorCount;
  generalDialDescriptorCount = packed.generalDialDescriptorCount;
  graphDialDescriptorCount = packed.graphDialDescriptorCount;
  maxHistogramChunksPerBin = packed.maxHistogramChunksPerBin;
  if( externalDialOccurrences.empty() ){ externalDialOccurrences.emplace_back(); }
  auto totalDynamicDialOccurrences = packed.totalDynamicDialOccurrences;
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

  std::vector<ResponseSmoothing<float>> smoothings(1);
  std::map<std::array<uint32_t, 6>, uint32_t> smoothingIndices;
  auto smoothingIndex = [&](const ResponseSmoothing<float>& smoothing_){
    if( smoothing_.minWidth == 0 and smoothing_.maxWidth == 0 ){ return uint32_t(0); }
    std::array<uint32_t, 6> key{};
    static_assert(sizeof(key) == sizeof(smoothing_));
    std::memcpy(key.data(), &smoothing_, sizeof(smoothing_));
    auto found = smoothingIndices.find(key);
    if( found != smoothingIndices.end() ){ return found->second; }
    LogThrowIf(smoothings.size() > (std::numeric_limits<uint32_t>::max() >> 2), "Too many MPS smoothing records.");
    auto index = uint32_t(smoothings.size());
    smoothings.push_back(smoothing_);
    smoothingIndices.emplace(key, index);
    return index;
  };
  auto uploadSplineDescriptors = [&](const auto& descriptors_){
    std::vector<MpsSplineDialDescriptor> reduced;
    reduced.reserve(descriptors_.size());
    for( const auto& descriptor : descriptors_ ){
      reduced.push_back({descriptor.parameterIndex, descriptor.splineOffset, descriptor.splineSize,
        descriptor.flags | (smoothingIndex(descriptor.smoothing) << 2), descriptor.minResponse, descriptor.maxResponse});
    }
    return makePrivateBuffer(device, commandQueue, reduced);
  };
  std::vector<MpsNormDialOccurrence> reducedNorms;
  reducedNorms.reserve(normDialOccurrences.size());
  for( const auto& occurrence : normDialOccurrences ){
    reducedNorms.push_back({occurrence.parameterIndex, occurrence.minResponse, occurrence.maxResponse,
                            smoothingIndex(occurrence.smoothing)});
  }
  reducedNormDialOccurrencesBuffer = makePrivateBuffer(device, commandQueue, reducedNorms);
  std::vector<MpsExternalDialOccurrence> reducedExternals;
  reducedExternals.reserve(externalDialOccurrences.size());
  for( const auto& occurrence : externalDialOccurrences ){
    reducedExternals.push_back({occurrence.weightIndex, occurrence.minResponse, occurrence.maxResponse,
                                smoothingIndex(occurrence.smoothing)});
  }
  reducedExternalDialOccurrencesBuffer = makePrivateBuffer(device, commandQueue, reducedExternals);
  reducedCompactDialDescriptorsBuffer = uploadSplineDescriptors(compactDialDescriptors);
  reducedUniformDialDescriptorsBuffer = uploadSplineDescriptors(uniformDialDescriptors);
  reducedMonotonicDialDescriptorsBuffer = uploadSplineDescriptors(monotonicDialDescriptors);
  reducedGeneralDialDescriptorsBuffer = uploadSplineDescriptors(generalDialDescriptors);
  reducedGraphDialDescriptorsBuffer = uploadSplineDescriptors(graphDialDescriptors);
  if( ordered ){
    orderedReducedCompactDialDescriptorsBuffer = uploadSplineDescriptors(ordered->compact);
    orderedReducedUniformDialDescriptorsBuffer = uploadSplineDescriptors(ordered->uniform);
    orderedReducedMonotonicDialDescriptorsBuffer = uploadSplineDescriptors(ordered->monotonic);
    orderedReducedGeneralDialDescriptorsBuffer = uploadSplineDescriptors(ordered->general);
    orderedReducedGraphDialDescriptorsBuffer = uploadSplineDescriptors(ordered->graph);
    orderedSplineDataBuffer = makePrivateBuffer(device, commandQueue, ordered->data);
  }
  responseSmoothingBuffer = makePrivateBuffer(device, commandQueue, smoothings);

  baseWeightsBuffer = makePrivateBuffer(device, commandQueue, baseWeights);
  eventDialRangesBuffer = makePrivateBuffer(device, commandQueue, eventDialRanges);
  if( enableDiagnosticDescriptorComparison ){ normDialOccurrencesBuffer = makePrivateBuffer(device, commandQueue, normDialOccurrences); }
  compactDialIndicesBuffer = makePrivateBuffer(device, commandQueue, compactDialIndices);
  uniformDialIndicesBuffer = makePrivateBuffer(device, commandQueue, uniformDialIndices);
  monotonicDialIndicesBuffer = makePrivateBuffer(device, commandQueue, monotonicDialIndices);
  generalDialIndicesBuffer = makePrivateBuffer(device, commandQueue, generalDialIndices);
  graphDialIndicesBuffer = makePrivateBuffer(device, commandQueue, graphDialIndices);
  if( enableDiagnosticDescriptorComparison ){ compactDialDescriptorsBuffer = makePrivateBuffer(device, commandQueue, compactDialDescriptors); }
  if( enableDiagnosticDescriptorComparison ){ uniformDialDescriptorsBuffer = makePrivateBuffer(device, commandQueue, uniformDialDescriptors); }
  if( enableDiagnosticDescriptorComparison ){ monotonicDialDescriptorsBuffer = makePrivateBuffer(device, commandQueue, monotonicDialDescriptors); }
  if( enableDiagnosticDescriptorComparison ){ generalDialDescriptorsBuffer = makePrivateBuffer(device, commandQueue, generalDialDescriptors); }
  if( enableDiagnosticDescriptorComparison ){ graphDialDescriptorsBuffer = makePrivateBuffer(device, commandQueue, graphDialDescriptors); }
  globalBinsBuffer = makePrivateBuffer(device, commandQueue, globalBins);
  binEventOffsetsBuffer = makePrivateBuffer(device, commandQueue, binEventOffsets);
  binEventIndicesBuffer = makePrivateBuffer(device, commandQueue, binEventIndices);
  splineDataBuffer = makePrivateBuffer(device, commandQueue, splineData);
  // Keep the existing response-slot numbering. Only the dispatch list is
  // compacted, so event references and floating-point operations are unchanged.
  auto uploadCachedIndices = [&](const auto& descriptors_, uint32_t descriptorCount_, uint32_t cachedCount_){
    std::vector<uint32_t> indices;
    indices.reserve(std::max(uint32_t(1), cachedCount_));
    for( uint32_t i = 0; i < descriptorCount_; ++i ){
      if( descriptors_[i].flags & kDeviceDialFlagCached ){ indices.emplace_back(i); }
    }
    LogThrowIf(indices.size() != cachedCount_, "Inconsistent MPS cached dial count.");
    if( indices.empty() ){ indices.emplace_back(0); } // Valid binding for empty/full-scan diagnostics.
    return makePrivateBuffer(device, commandQueue, indices);
  };
  compactCachedDialIndicesBuffer = uploadCachedIndices(compactDialDescriptors, compactDialDescriptorCount, compactCachedDialCount);
  uniformCachedDialIndicesBuffer = uploadCachedIndices(uniformDialDescriptors, uniformDialDescriptorCount, uniformCachedDialCount);
  monotonicCachedDialIndicesBuffer = uploadCachedIndices(monotonicDialDescriptors, monotonicDialDescriptorCount, monotonicCachedDialCount);
  generalCachedDialIndicesBuffer = uploadCachedIndices(generalDialDescriptors, generalDialDescriptorCount, generalCachedDialCount);
  graphCachedDialIndicesBuffer = uploadCachedIndices(graphDialDescriptors, graphDialDescriptorCount, graphCachedDialCount);
  const auto cachedResponseBufferBytes = [this](std::size_t descriptorCount_){
    return (enableCachedDialResponses ? descriptorCount_ : std::size_t(1)) * sizeof(float);
  };
  compactCachedResponsesBuffer = makePrivateEmptyBuffer(device, cachedResponseBufferBytes(compactDialDescriptors.size()));
  uniformCachedResponsesBuffer = makePrivateEmptyBuffer(device, cachedResponseBufferBytes(uniformDialDescriptors.size()));
  monotonicCachedResponsesBuffer = makePrivateEmptyBuffer(device, cachedResponseBufferBytes(monotonicDialDescriptors.size()));
  generalCachedResponsesBuffer = makePrivateEmptyBuffer(device, cachedResponseBufferBytes(generalDialDescriptors.size()));
  graphCachedResponsesBuffer = makePrivateEmptyBuffer(device, cachedResponseBufferBytes(graphDialDescriptors.size()));
  eventWeightsBuffer = makePrivateEmptyBuffer(device, model.events.size() * sizeof(float));
  eventWeightsReadbackBuffer = makeSharedEmptyBuffer(device, model.events.size() * sizeof(float));
  std::size_t externalWeightCount{0};
  for( const auto& block : model.externalWeightBlocks ){
    externalWeightCount = std::max(externalWeightCount, block.offset + block.count);
  }
  externalWeightsBuffer = makeSharedEmptyBuffer(device, std::max<std::size_t>(1, externalWeightCount) * sizeof(float));
  if( enableDiagnosticDescriptorComparison ){ externalDialOccurrencesBuffer = makePrivateBuffer(device, commandQueue, externalDialOccurrences); }
  externalWeightGenerations.assign(model.externalWeightBlocks.size(), 0);
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

  if( enableDiagnosticPayloadComparison and
      (orderedSplineDataBuffer == nil or orderedReducedCompactDialDescriptorsBuffer == nil or orderedReducedUniformDialDescriptorsBuffer == nil
       or orderedReducedMonotonicDialDescriptorsBuffer == nil or orderedReducedGeneralDialDescriptorsBuffer == nil or orderedReducedGraphDialDescriptorsBuffer == nil) ){
    releaseDeviceBuffers();
    return fail("Ordered MPS payload allocation failed.");
  }
  if( reducedNormDialOccurrencesBuffer == nil or reducedExternalDialOccurrencesBuffer == nil or reducedCompactDialDescriptorsBuffer == nil or reducedUniformDialDescriptorsBuffer == nil or reducedMonotonicDialDescriptorsBuffer == nil or reducedGeneralDialDescriptorsBuffer == nil or reducedGraphDialDescriptorsBuffer == nil
      or responseSmoothingBuffer == nil
      or compactCachedDialIndicesBuffer == nil or uniformCachedDialIndicesBuffer == nil or monotonicCachedDialIndicesBuffer == nil or generalCachedDialIndicesBuffer == nil or graphCachedDialIndicesBuffer == nil
      or externalWeightsBuffer == nil or (enableDiagnosticDescriptorComparison and externalDialOccurrencesBuffer == nil) or baseWeightsBuffer == nil or eventDialRangesBuffer == nil or (enableDiagnosticDescriptorComparison and normDialOccurrencesBuffer == nil)
      or compactDialIndicesBuffer == nil or uniformDialIndicesBuffer == nil
      or monotonicDialIndicesBuffer == nil or generalDialIndicesBuffer == nil
      or graphDialIndicesBuffer == nil
      or (enableDiagnosticDescriptorComparison and compactDialDescriptorsBuffer == nil) or (enableDiagnosticDescriptorComparison and uniformDialDescriptorsBuffer == nil)
      or (enableDiagnosticDescriptorComparison and monotonicDialDescriptorsBuffer == nil) or (enableDiagnosticDescriptorComparison and generalDialDescriptorsBuffer == nil)
      or (enableDiagnosticDescriptorComparison and graphDialDescriptorsBuffer == nil)
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

  isWeightModelSupported = true;
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
