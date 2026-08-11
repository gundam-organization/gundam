#include "MpsBackendInternal.h"

Backends::MpsBackendImpl::MpsBackendImpl() : model(engineView.propagation), likelihoodModel(engineView.likelihood) {
  device = MTLCreateSystemDefaultDevice();
  if( device == nil ){ return; }

  NSError* error = nil;
  auto* compileOptions = [[MTLCompileOptions alloc] init];
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
  if( @available(macOS 15.0, *) ){
    compileOptions.mathMode = MTLMathModeSafe;
  }
  else{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    compileOptions.fastMathEnabled = NO;
#pragma clang diagnostic pop
  }
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  compileOptions.fastMathEnabled = NO;
#pragma clang diagnostic pop
#endif
  id<MTLLibrary> library = [device newLibraryWithSource:kMpsBackendMetalSource options:compileOptions error:&error];
  [compileOptions release];
  if( library == nil ){
    if( error != nil ){
      LogError << "Could not compile MPS backend Metal library: "
               << [[error localizedDescription] UTF8String] << std::endl;
    }
    return;
  }

  eventWeightsPipeline = makePipeline(device, library, @"compute_event_weights");
  cachedCompactResponsesPipeline = makePipeline(device, library, @"compute_cached_compact_responses");
  cachedUniformResponsesPipeline = makePipeline(device, library, @"compute_cached_uniform_responses");
  cachedMonotonicResponsesPipeline = makePipeline(device, library, @"compute_cached_monotonic_responses");
  cachedGeneralResponsesPipeline = makePipeline(device, library, @"compute_cached_general_responses");
  cachedGraphResponsesPipeline = makePipeline(device, library, @"compute_cached_graph_responses");
  histogramPipeline = makePipeline(device, library, @"fill_histograms");
  histogramPartialsPipeline = makePipeline(device, library, @"fill_histogram_partials_by_bin");
  histogramFinalizePipeline = makePipeline(device, library, @"finalize_histograms_from_partials");
  [library release];
  if( eventWeightsPipeline == nil or cachedCompactResponsesPipeline == nil or cachedUniformResponsesPipeline == nil
      or cachedMonotonicResponsesPipeline == nil or cachedGeneralResponsesPipeline == nil
      or cachedGraphResponsesPipeline == nil or histogramPipeline == nil
      or histogramPartialsPipeline == nil or histogramFinalizePipeline == nil ){ return; }

  commandQueue = [device newCommandQueue];
  if( commandQueue == nil ){ return; }

  isAvailable = true;
}

Backends::MpsBackendImpl::~MpsBackendImpl() {
  releaseDeviceBuffers();
  [eventWeightsPipeline release];
  [cachedCompactResponsesPipeline release];
  [cachedUniformResponsesPipeline release];
  [cachedMonotonicResponsesPipeline release];
  [cachedGeneralResponsesPipeline release];
  [cachedGraphResponsesPipeline release];
  [histogramPipeline release];
  [histogramPartialsPipeline release];
  [histogramFinalizePipeline release];
  [commandQueue release];
  [device release];
}

void Backends::MpsBackendImpl::releaseDeviceBuffers() {
  releaseBuffer(eventWeightsBuffer);
  releaseBuffer(baseWeightsBuffer);
  releaseBuffer(eventDialRangesBuffer);
  releaseBuffer(normDialOccurrencesBuffer);
  releaseBuffer(compactDialIndicesBuffer);
  releaseBuffer(uniformDialIndicesBuffer);
  releaseBuffer(monotonicDialIndicesBuffer);
  releaseBuffer(generalDialIndicesBuffer);
  releaseBuffer(graphDialIndicesBuffer);
  releaseBuffer(compactDialDescriptorsBuffer);
  releaseBuffer(uniformDialDescriptorsBuffer);
  releaseBuffer(monotonicDialDescriptorsBuffer);
  releaseBuffer(generalDialDescriptorsBuffer);
  releaseBuffer(graphDialDescriptorsBuffer);
  releaseBuffer(compactCachedResponsesBuffer);
  releaseBuffer(uniformCachedResponsesBuffer);
  releaseBuffer(monotonicCachedResponsesBuffer);
  releaseBuffer(generalCachedResponsesBuffer);
  releaseBuffer(graphCachedResponsesBuffer);
  releaseBuffer(globalBinsBuffer);
  releaseBuffer(binEventOffsetsBuffer);
  releaseBuffer(binEventIndicesBuffer);
  releaseBuffer(splineDataBuffer);
  releaseBuffer(parametersBuffer);
  releaseBuffer(partialHistSumsBuffer);
  releaseBuffer(partialHistSumSquaresBuffer);
  releaseBuffer(histSumsBuffer);
  releaseBuffer(histSumSquaresBuffer);
  releaseBuffer(eventWeightsReadbackBuffer);
  releaseBuffer(histSumsReadbackBuffer);
  releaseBuffer(histSumSquaresReadbackBuffer);
  releaseBuffer(nEventsBuffer);
  releaseBuffer(totalBinsBuffer);
  releaseBuffer(maxHistogramChunksPerBinBuffer);
  releaseBuffer(histogramChunkSizeBuffer);
}

bool Backends::MpsBackendImpl::isCurrentToken(const PropagationToken& token_) const {
  return token_.isValid and lastResult.token.isValid and token_.id == lastResult.token.id;
}

void Backends::MpsBackendImpl::resetResult() {
  lastResult.token.id = nextTokenId++;
  lastResult.token.isValid = true;
  lastResult.status = PropagationStatus();
  lastResult.status.backend = BackendStatus::Running;
  lastResult.eventWeights.clear();
  lastResult.histSums.clear();
  lastResult.histSumSquares.clear();
  lastResult.likelihood = 0;
  lastTiming = buildTiming;
  lastResult.status.eventWeights = OutputState::Scheduled;
  lastResult.status.histograms = OutputState::Scheduled;
  lastResult.status.sampleLikelihoods = OutputState::Scheduled;
  lastResult.status.statLikelihood = OutputState::Scheduled;
}
