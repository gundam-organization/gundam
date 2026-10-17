#include "MpsBackendInternal.h"
#include "DeviceWeightModel.h"

bool Backends::MpsBackendImpl::buildWeightModel() {
  releaseDeviceBuffers();
  isWeightModelSupported = false;
  deviceModelFallbackReason.clear();
  if( not isAvailable ){ deviceModelFallbackReason = "Metal is not available."; return false; }
  DeviceWeightModel packed;
  if( not packDeviceWeightModel(model, enableCachedDialResponses, packed, buildTiming,
                                  deviceModelFallbackReason, "MPS backend") ){ return false; }
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
  externalDialOccurrencesBuffer = makePrivateBuffer(device, commandQueue, externalDialOccurrences);
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

  if( externalWeightsBuffer == nil or externalDialOccurrencesBuffer == nil or baseWeightsBuffer == nil or eventDialRangesBuffer == nil or normDialOccurrencesBuffer == nil
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
