#include "MpsBackendInternal.h"

#include "Semantics/BackendHostPropagation.h"

void Backends::MpsBackendImpl::updateDeviceParameters(const ParameterSnapshot& parameters_) {
  auto start = std::chrono::steady_clock::now();
  if( parameterValuesScratch.size() != model.parameterCount ){
    parameterValuesScratch.resize(model.parameterCount);
  }
  LogThrowIf(parameters_.empty(), "MpsBackend requires a populated ParameterSnapshot.");
  LogThrowIf(parameters_.values.size() != model.parameterCount,
             "ParameterSnapshot size mismatch: " << parameters_.values.size()
                                                 << " != " << model.parameterCount);
  for( std::size_t iPar = 0 ; iPar < model.parameterCount ; iPar++ ){
    parameterValuesScratch[iPar] = float(parameters_.values.at(iPar));
  }
  copyToBuffer(parametersBuffer, parameterValuesScratch);
  lastTiming.parameterUploadSeconds += secondsSince(start);
}

bool Backends::MpsBackendImpl::encodeEventWeights(id<MTLComputeCommandEncoder> encoder) {
  if( not isDeviceModelSupported ){ return false; }
  [encoder setComputePipelineState:eventWeightsPipeline];
  [encoder setBuffer:eventWeightsBuffer offset:0 atIndex:0];
  [encoder setBuffer:baseWeightsBuffer offset:0 atIndex:1];
  [encoder setBuffer:eventDialRangesBuffer offset:0 atIndex:2];
  [encoder setBuffer:normDialOccurrencesBuffer offset:0 atIndex:3];
  [encoder setBuffer:parametersBuffer offset:0 atIndex:4];
  [encoder setBuffer:compactDialIndicesBuffer offset:0 atIndex:5];
  [encoder setBuffer:compactDialDescriptorsBuffer offset:0 atIndex:6];
  [encoder setBuffer:compactCachedResponsesBuffer offset:0 atIndex:7];
  [encoder setBuffer:uniformDialIndicesBuffer offset:0 atIndex:8];
  [encoder setBuffer:uniformDialDescriptorsBuffer offset:0 atIndex:9];
  [encoder setBuffer:uniformCachedResponsesBuffer offset:0 atIndex:10];
  [encoder setBuffer:monotonicDialIndicesBuffer offset:0 atIndex:11];
  [encoder setBuffer:monotonicDialDescriptorsBuffer offset:0 atIndex:12];
  [encoder setBuffer:monotonicCachedResponsesBuffer offset:0 atIndex:13];
  [encoder setBuffer:generalDialIndicesBuffer offset:0 atIndex:14];
  [encoder setBuffer:generalDialDescriptorsBuffer offset:0 atIndex:15];
  [encoder setBuffer:generalCachedResponsesBuffer offset:0 atIndex:16];
  [encoder setBuffer:graphDialIndicesBuffer offset:0 atIndex:17];
  [encoder setBuffer:graphDialDescriptorsBuffer offset:0 atIndex:18];
  [encoder setBuffer:graphCachedResponsesBuffer offset:0 atIndex:19];
  [encoder setBuffer:splineDataBuffer offset:0 atIndex:20];
  [encoder setBuffer:nEventsBuffer offset:0 atIndex:21];

  NSUInteger width = std::min<NSUInteger>(eventWeightsPipeline.maxTotalThreadsPerThreadgroup, 256);
  if( width == 0 ){ width = 1; }
  [encoder dispatchThreads:MTLSizeMake(NSUInteger(model.events.size()), 1, 1)
     threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  return true;
}

bool Backends::MpsBackendImpl::encodeCachedDialResponses(id<MTLComputeCommandEncoder> encoder,
                                                         id<MTLComputePipelineState> pipeline_,
                                                         id<MTLBuffer> cachedResponsesBuffer_,
                                                         id<MTLBuffer> descriptorsBuffer_,
                                                         uint32_t descriptorCount_) {
  if( not isDeviceModelSupported ){ return false; }
  if( descriptorCount_ == 0 ){ return true; }
  [encoder setComputePipelineState:pipeline_];
  [encoder setBuffer:cachedResponsesBuffer_ offset:0 atIndex:0];
  [encoder setBuffer:descriptorsBuffer_ offset:0 atIndex:1];
  [encoder setBuffer:splineDataBuffer offset:0 atIndex:2];
  [encoder setBuffer:parametersBuffer offset:0 atIndex:3];
  [encoder setBytes:&descriptorCount_ length:sizeof(descriptorCount_) atIndex:4];

  NSUInteger width = std::min<NSUInteger>(pipeline_.maxTotalThreadsPerThreadgroup, 256);
  if( width == 0 ){ width = 1; }
  [encoder dispatchThreads:MTLSizeMake(NSUInteger(descriptorCount_), 1, 1)
     threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  return true;
}

bool Backends::MpsBackendImpl::encodeHistogramsFromDeviceWeights(id<MTLComputeCommandEncoder> encoder) {
  if( not isDeviceModelSupported ){ return false; }
  [encoder setComputePipelineState:histogramPartialsPipeline];
  [encoder setBuffer:eventWeightsBuffer offset:0 atIndex:0];
  [encoder setBuffer:binEventOffsetsBuffer offset:0 atIndex:1];
  [encoder setBuffer:binEventIndicesBuffer offset:0 atIndex:2];
  [encoder setBuffer:partialHistSumsBuffer offset:0 atIndex:3];
  [encoder setBuffer:partialHistSumSquaresBuffer offset:0 atIndex:4];
  [encoder setBuffer:totalBinsBuffer offset:0 atIndex:5];
  [encoder setBuffer:maxHistogramChunksPerBinBuffer offset:0 atIndex:6];
  [encoder setBuffer:histogramChunkSizeBuffer offset:0 atIndex:7];

  NSUInteger width = std::min<NSUInteger>(histogramPartialsPipeline.maxTotalThreadsPerThreadgroup, 256);
  if( width == 0 ){ width = 1; }
  [encoder dispatchThreads:MTLSizeMake(NSUInteger(model.totalBins) * maxHistogramChunksPerBin, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];

  [encoder setComputePipelineState:histogramFinalizePipeline];
  [encoder setBuffer:partialHistSumsBuffer offset:0 atIndex:0];
  [encoder setBuffer:partialHistSumSquaresBuffer offset:0 atIndex:1];
  [encoder setBuffer:histSumsBuffer offset:0 atIndex:2];
  [encoder setBuffer:histSumSquaresBuffer offset:0 atIndex:3];
  [encoder setBuffer:totalBinsBuffer offset:0 atIndex:4];
  [encoder setBuffer:maxHistogramChunksPerBinBuffer offset:0 atIndex:5];

  width = std::min<NSUInteger>(histogramFinalizePipeline.maxTotalThreadsPerThreadgroup, 256);
  if( width == 0 ){ width = 1; }
  [encoder dispatchThreads:MTLSizeMake(NSUInteger(model.totalBins), 1, 1)
     threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  return true;
}

bool Backends::MpsBackendImpl::runDevicePropagation(const ParameterSnapshot& parameters_, bool needHistograms_) {
  if( not isDeviceModelSupported ){ return false; }
  updateDeviceParameters(parameters_);

  auto encodeAllCachedResponses = [&](id<MTLComputeCommandEncoder> encoder_) {
    return encodeCachedDialResponses(encoder_, cachedCompactResponsesPipeline, compactCachedResponsesBuffer, compactDialDescriptorsBuffer, compactDialDescriptorCount)
           and encodeCachedDialResponses(encoder_, cachedUniformResponsesPipeline, uniformCachedResponsesBuffer, uniformDialDescriptorsBuffer, uniformDialDescriptorCount)
           and encodeCachedDialResponses(encoder_, cachedMonotonicResponsesPipeline, monotonicCachedResponsesBuffer, monotonicDialDescriptorsBuffer, monotonicDialDescriptorCount)
           and encodeCachedDialResponses(encoder_, cachedGeneralResponsesPipeline, generalCachedResponsesBuffer, generalDialDescriptorsBuffer, generalDialDescriptorCount)
           and encodeCachedDialResponses(encoder_, cachedGraphResponsesPipeline, graphCachedResponsesBuffer, graphDialDescriptorsBuffer, graphDialDescriptorCount);
  };

  if( GundamGlobals::isDebug() ){
    if( cachedDialCount > 0 ){
      auto stageStart = std::chrono::steady_clock::now();
      auto encodeStart = std::chrono::steady_clock::now();
      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      if( not encodeAllCachedResponses(encoder) ){
        [encoder endEncoding];
        return false;
      }
      [encoder endEncoding];
      lastTiming.commandEncodeSeconds += secondsSince(encodeStart);
      auto waitStart = std::chrono::steady_clock::now();
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      lastTiming.deviceWaitSeconds += secondsSince(waitStart);
      if( commandBuffer.status != MTLCommandBufferStatusCompleted ){
        return false;
      }
      lastTiming.cachedDialStageSeconds += secondsSince(stageStart);
    }

    {
      auto stageStart = std::chrono::steady_clock::now();
      auto encodeStart = std::chrono::steady_clock::now();
      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      if( not encodeEventWeights(encoder) ){
        [encoder endEncoding];
        return false;
      }
      [encoder endEncoding];
      lastTiming.commandEncodeSeconds += secondsSince(encodeStart);
      auto waitStart = std::chrono::steady_clock::now();
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      lastTiming.deviceWaitSeconds += secondsSince(waitStart);
      if( commandBuffer.status != MTLCommandBufferStatusCompleted ){
        return false;
      }
      lastTiming.eventWeightsStageSeconds += secondsSince(stageStart);
    }

    if( needHistograms_ ){
      auto stageStart = std::chrono::steady_clock::now();
      auto encodeStart = std::chrono::steady_clock::now();
      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      if( not encodeHistogramsFromDeviceWeights(encoder) ){
        [encoder endEncoding];
        return false;
      }
      [encoder endEncoding];
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
      std::size_t histogramBytes = std::size_t(model.totalBins) * sizeof(float);
      [blitEncoder copyFromBuffer:histSumsBuffer sourceOffset:0 toBuffer:histSumsReadbackBuffer destinationOffset:0 size:histogramBytes];
      [blitEncoder copyFromBuffer:histSumSquaresBuffer sourceOffset:0 toBuffer:histSumSquaresReadbackBuffer destinationOffset:0 size:histogramBytes];
      [blitEncoder endEncoding];
      lastTiming.histogramReadbackBytes += 2 * histogramBytes;
      lastTiming.commandEncodeSeconds += secondsSince(encodeStart);
      auto waitStart = std::chrono::steady_clock::now();
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      lastTiming.deviceWaitSeconds += secondsSince(waitStart);
      if( commandBuffer.status != MTLCommandBufferStatusCompleted ){
        return false;
      }
      lastTiming.histogramStageSeconds += secondsSince(stageStart);

      auto readbackStart = std::chrono::steady_clock::now();
      auto* histSums = static_cast<float*>(histSumsReadbackBuffer.contents);
      auto* histSumSquares = static_cast<float*>(histSumSquaresReadbackBuffer.contents);
      lastResult.histSums.resize(model.totalBins);
      lastResult.histSumSquares.resize(model.totalBins);
      for( int iBin = 0 ; iBin < model.totalBins ; iBin++ ){
        lastResult.histSums[iBin] = histSums[iBin];
        lastResult.histSumSquares[iBin] = histSumSquares[iBin];
      }
      lastTiming.histogramReadbackSeconds += secondsSince(readbackStart);
    }

    return true;
  }

  auto encodeStart = std::chrono::steady_clock::now();
  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if( not encodeAllCachedResponses(encoder) ){
    [encoder endEncoding];
    return false;
  }
  if( not encodeEventWeights(encoder) ){
    [encoder endEncoding];
    return false;
  }
  if( needHistograms_ and not encodeHistogramsFromDeviceWeights(encoder) ){
    [encoder endEncoding];
    return false;
  }
  [encoder endEncoding];
  if( needHistograms_ ){
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    std::size_t histogramBytes = std::size_t(model.totalBins) * sizeof(float);
    [blitEncoder copyFromBuffer:histSumsBuffer sourceOffset:0 toBuffer:histSumsReadbackBuffer destinationOffset:0 size:histogramBytes];
    [blitEncoder copyFromBuffer:histSumSquaresBuffer sourceOffset:0 toBuffer:histSumSquaresReadbackBuffer destinationOffset:0 size:histogramBytes];
    [blitEncoder endEncoding];
    lastTiming.histogramReadbackBytes += 2 * histogramBytes;
  }
  [commandBuffer commit];
  lastTiming.commandEncodeSeconds += secondsSince(encodeStart);
  auto waitStart = std::chrono::steady_clock::now();
  [commandBuffer waitUntilCompleted];
  lastTiming.deviceWaitSeconds += secondsSince(waitStart);

  if( commandBuffer.status != MTLCommandBufferStatusCompleted ){
    return false;
  }

  if( needHistograms_ ){
    auto readbackStart = std::chrono::steady_clock::now();
    auto* histSums = static_cast<float*>(histSumsReadbackBuffer.contents);
    auto* histSumSquares = static_cast<float*>(histSumSquaresReadbackBuffer.contents);
    lastResult.histSums.resize(model.totalBins);
    lastResult.histSumSquares.resize(model.totalBins);
    for( int iBin = 0 ; iBin < model.totalBins ; iBin++ ){
      lastResult.histSums[iBin] = histSums[iBin];
      lastResult.histSumSquares[iBin] = histSumSquares[iBin];
    }
    lastTiming.histogramReadbackSeconds += secondsSince(readbackStart);
  }

  return true;
}

void Backends::MpsBackendImpl::copyDeviceEventWeightsToHostResult() {
  LogThrowIf(eventWeightsBuffer == nil);
  auto start = std::chrono::steady_clock::now();
  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
  std::size_t byteSize = model.events.size() * sizeof(float);
  [encoder copyFromBuffer:eventWeightsBuffer sourceOffset:0 toBuffer:eventWeightsReadbackBuffer destinationOffset:0 size:byteSize];
  [encoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  LogThrowIf(commandBuffer.status != MTLCommandBufferStatusCompleted,
             "Could not copy MPS event weights back to host.");
  lastTiming.eventWeightReadbackBytes += byteSize;

  auto* weights = static_cast<float*>(eventWeightsReadbackBuffer.contents);
  lastResult.eventWeights.resize(model.events.size());
  for( const auto& event : model.events ){
    lastResult.eventWeights[event.resultIndex] = weights[event.resultIndex];
  }
  lastTiming.eventWeightReadbackSeconds += secondsSince(start);
}

bool Backends::MpsBackendImpl::calculateHistogramsOnDevice() {
  if( not isAvailable ){ return false; }
  if( model.totalBins <= 0 ){ return false; }
  if( model.events.empty() ){
    lastResult.histSums.assign(model.totalBins, 0);
    lastResult.histSumSquares.assign(model.totalBins, 0);
    return true;
  }

  LogThrowIf(lastResult.eventWeights.size() != model.events.size(),
             "MPS histogram fallback requires precomputed event weights.");

  std::vector<float> eventWeightsFloat(lastResult.eventWeights.size());
  std::vector<int> globalBins(model.events.size());
  for( const auto& event : model.events ){
    eventWeightsFloat[event.resultIndex] = float(lastResult.eventWeights[event.resultIndex]);
    globalBins[event.resultIndex] = event.globalBinIndex;
  }

  auto eventWeightsBufferLocal = makeSharedBuffer(device, eventWeightsFloat);
  auto globalBinsBufferLocal = makeSharedBuffer(device, globalBins);
  auto histSumsBufferLocal = [device newBufferWithLength:std::size_t(model.totalBins) * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
  auto histSumSquaresBufferLocal = [device newBufferWithLength:std::size_t(model.totalBins) * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
  uint32_t nEvents = uint32_t(model.events.size());
  uint32_t totalBins = uint32_t(model.totalBins);
  auto nEventsBufferLocal = [device newBufferWithBytes:&nEvents length:sizeof(nEvents) options:MTLResourceStorageModeShared];
  auto totalBinsBufferLocal = [device newBufferWithBytes:&totalBins length:sizeof(totalBins) options:MTLResourceStorageModeShared];

  if( eventWeightsBufferLocal == nil or globalBinsBufferLocal == nil or histSumsBufferLocal == nil
      or histSumSquaresBufferLocal == nil or nEventsBufferLocal == nil or totalBinsBufferLocal == nil ){
    [eventWeightsBufferLocal release];
    [globalBinsBufferLocal release];
    [histSumsBufferLocal release];
    [histSumSquaresBufferLocal release];
    [nEventsBufferLocal release];
    [totalBinsBufferLocal release];
    return false;
  }

  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  [encoder setComputePipelineState:histogramPipeline];
  [encoder setBuffer:eventWeightsBufferLocal offset:0 atIndex:0];
  [encoder setBuffer:globalBinsBufferLocal offset:0 atIndex:1];
  [encoder setBuffer:histSumsBufferLocal offset:0 atIndex:2];
  [encoder setBuffer:histSumSquaresBufferLocal offset:0 atIndex:3];
  [encoder setBuffer:nEventsBufferLocal offset:0 atIndex:4];
  [encoder setBuffer:totalBinsBufferLocal offset:0 atIndex:5];

  NSUInteger width = std::min<NSUInteger>(histogramPipeline.maxTotalThreadsPerThreadgroup, 256);
  if( width == 0 ){ width = 1; }
  MTLSize threadsPerThreadgroup = MTLSizeMake(width, 1, 1);
  MTLSize threadsPerGrid = MTLSizeMake(NSUInteger(model.totalBins), 1, 1);
  [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
  [encoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  bool ok = commandBuffer.status == MTLCommandBufferStatusCompleted;
  if( ok ){
    auto* histSums = static_cast<float*>(histSumsBufferLocal.contents);
    auto* histSumSquares = static_cast<float*>(histSumSquaresBufferLocal.contents);
    lastResult.histSums.resize(model.totalBins);
    lastResult.histSumSquares.resize(model.totalBins);
    for( int iBin = 0 ; iBin < model.totalBins ; iBin++ ){
      lastResult.histSums[iBin] = histSums[iBin];
      lastResult.histSumSquares[iBin] = histSumSquares[iBin];
    }
  }

  [eventWeightsBufferLocal release];
  [globalBinsBufferLocal release];
  [histSumsBufferLocal release];
  [histSumSquaresBufferLocal release];
  [nEventsBufferLocal release];
  [totalBinsBufferLocal release];
  return ok;
}

void Backends::MpsBackendImpl::calculateLikelihood() {
  auto start = std::chrono::steady_clock::now();
  lastResult.likelihood = Semantics::calculateLikelihood(
      likelihoodModel,
      lastResult.histSums,
      lastResult.histSumSquares
  );
  lastTiming.likelihoodHostSeconds += secondsSince(start);
}

void Backends::MpsBackendImpl::materializeEventWeights() {
  auto start = std::chrono::steady_clock::now();
  if( lastResult.eventWeights.empty() and eventWeightsBuffer != nil ){
    copyDeviceEventWeightsToHostResult();
  }
  LogThrowIf(lastResult.eventWeights.size() != model.events.size());
  lastTiming.eventWeightMaterializationSeconds += secondsSince(start);
}

void Backends::MpsBackendImpl::materializeHistograms() {
  auto start = std::chrono::steady_clock::now();
  LogThrowIf(lastResult.histSums.size() != std::size_t(model.totalBins));
  LogThrowIf(lastResult.histSumSquares.size() != std::size_t(model.totalBins));
  lastTiming.histogramMaterializationSeconds += secondsSince(start);
}
