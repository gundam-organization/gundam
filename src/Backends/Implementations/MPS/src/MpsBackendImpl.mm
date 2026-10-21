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
  reducedEventWeightsPipeline = makePipeline(device, library, @"compute_event_weights_reduced");
  cachedCompactResponsesPipeline = makePipeline(device, library, @"compute_cached_compact_responses");
  reducedCachedCompactResponsesPipeline = makePipeline(device, library, @"compute_cached_compact_responses_reduced");
  cachedUniformResponsesPipeline = makePipeline(device, library, @"compute_cached_uniform_responses");
  reducedCachedUniformResponsesPipeline = makePipeline(device, library, @"compute_cached_uniform_responses_reduced");
  cachedMonotonicResponsesPipeline = makePipeline(device, library, @"compute_cached_monotonic_responses");
  reducedCachedMonotonicResponsesPipeline = makePipeline(device, library, @"compute_cached_monotonic_responses_reduced");
  cachedGeneralResponsesPipeline = makePipeline(device, library, @"compute_cached_general_responses");
  reducedCachedGeneralResponsesPipeline = makePipeline(device, library, @"compute_cached_general_responses_reduced");
  cachedGraphResponsesPipeline = makePipeline(device, library, @"compute_cached_graph_responses");
  reducedCachedGraphResponsesPipeline = makePipeline(device, library, @"compute_cached_graph_responses_reduced");
  histogramPipeline = makePipeline(device, library, @"fill_histograms");
  histogramPartialsPipeline = makePipeline(device, library, @"fill_histogram_partials_by_bin");
  histogramFinalizePipeline = makePipeline(device, library, @"finalize_histograms_from_partials");
  if( eventWeightsPipeline == nil or cachedCompactResponsesPipeline == nil or cachedUniformResponsesPipeline == nil
      or cachedMonotonicResponsesPipeline == nil or cachedGeneralResponsesPipeline == nil
      or cachedGraphResponsesPipeline == nil or histogramPipeline == nil
      or histogramPartialsPipeline == nil or histogramFinalizePipeline == nil ){ [library release]; return; }

  if( reducedEventWeightsPipeline == nil or reducedCachedCompactResponsesPipeline == nil or reducedCachedUniformResponsesPipeline == nil or reducedCachedMonotonicResponsesPipeline == nil or reducedCachedGeneralResponsesPipeline == nil or reducedCachedGraphResponsesPipeline == nil ){ [library release]; return; }

  commandQueue = [device newCommandQueue];
  if( commandQueue == nil ){ [library release]; return; }
  eventPropagation = std::make_unique<MpsEventPropagation>(device, commandQueue, library);
  [library release];

  isAvailable = true;
}

Backends::MpsBackendImpl::~MpsBackendImpl() {
  eventPropagation.reset();
  releaseDeviceBuffers();
  [eventWeightsPipeline release];
  [reducedEventWeightsPipeline release];
  [cachedCompactResponsesPipeline release];
  [reducedCachedCompactResponsesPipeline release];
  [cachedUniformResponsesPipeline release];
  [reducedCachedUniformResponsesPipeline release];
  [cachedMonotonicResponsesPipeline release];
  [reducedCachedMonotonicResponsesPipeline release];
  [cachedGeneralResponsesPipeline release];
  [reducedCachedGeneralResponsesPipeline release];
  [cachedGraphResponsesPipeline release];
  [reducedCachedGraphResponsesPipeline release];
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
  releaseBuffer(reducedNormDialOccurrencesBuffer);
  releaseBuffer(compactDialIndicesBuffer);
  releaseBuffer(uniformDialIndicesBuffer);
  releaseBuffer(monotonicDialIndicesBuffer);
  releaseBuffer(generalDialIndicesBuffer);
  releaseBuffer(graphDialIndicesBuffer);
  releaseBuffer(compactDialDescriptorsBuffer);
  releaseBuffer(reducedCompactDialDescriptorsBuffer);
  releaseBuffer(orderedReducedCompactDialDescriptorsBuffer);
  releaseBuffer(uniformDialDescriptorsBuffer);
  releaseBuffer(reducedUniformDialDescriptorsBuffer);
  releaseBuffer(orderedReducedUniformDialDescriptorsBuffer);
  releaseBuffer(monotonicDialDescriptorsBuffer);
  releaseBuffer(reducedMonotonicDialDescriptorsBuffer);
  releaseBuffer(orderedReducedMonotonicDialDescriptorsBuffer);
  releaseBuffer(generalDialDescriptorsBuffer);
  releaseBuffer(reducedGeneralDialDescriptorsBuffer);
  releaseBuffer(orderedReducedGeneralDialDescriptorsBuffer);
  releaseBuffer(graphDialDescriptorsBuffer);
  releaseBuffer(reducedGraphDialDescriptorsBuffer);
  releaseBuffer(orderedReducedGraphDialDescriptorsBuffer);
  releaseBuffer(compactCachedResponsesBuffer);
  releaseBuffer(compactCachedDialIndicesBuffer);
  releaseBuffer(uniformCachedResponsesBuffer);
  releaseBuffer(uniformCachedDialIndicesBuffer);
  releaseBuffer(monotonicCachedResponsesBuffer);
  releaseBuffer(monotonicCachedDialIndicesBuffer);
  releaseBuffer(generalCachedResponsesBuffer);
  releaseBuffer(generalCachedDialIndicesBuffer);
  releaseBuffer(graphCachedResponsesBuffer);
  releaseBuffer(graphCachedDialIndicesBuffer);
  releaseBuffer(globalBinsBuffer);
  releaseBuffer(binEventOffsetsBuffer);
  releaseBuffer(binEventIndicesBuffer);
  releaseBuffer(splineDataBuffer);
  releaseBuffer(orderedSplineDataBuffer);
  releaseBuffer(responseSmoothingBuffer);
  releaseBuffer(parametersBuffer);
  releaseBuffer(externalWeightsBuffer);
  releaseBuffer(externalDialOccurrencesBuffer);
  releaseBuffer(reducedExternalDialOccurrencesBuffer);
  externalWeightGenerations.clear();
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
  lastResult.outsideSums.clear();
  lastResult.outsideSumSquares.clear();
  lastResult.likelihood = 0;
  lastTiming = buildTiming;
  lastResult.status.eventWeights = OutputState::Scheduled;
  lastResult.status.histograms = OutputState::Scheduled;
  lastResult.status.sampleLikelihoods = OutputState::Scheduled;
  lastResult.status.statLikelihood = OutputState::Scheduled;
}
