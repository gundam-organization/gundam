#include "MpsBackend.h"

#include "DialInputBuffer.h"
#include "DialInterface.h"
#include "DialResponseSupervisor.h"
#include "Event.h"
#include "Histogram.h"
#include "Norm.h"
#include "Parameter.h"

#include "Logger.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {
  static NSString* const kMpsBackendMetalSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void compute_norm_event_weights(
    device float* eventWeights [[buffer(0)]],
    device const float* baseWeights [[buffer(1)]],
    device const uint* eventDialOffsets [[buffer(2)]],
    device const uint* eventDialCounts [[buffer(3)]],
    device const uint* normParameterIndices [[buffer(4)]],
    device const float* normMinResponses [[buffer(5)]],
    device const float* normMaxResponses [[buffer(6)]],
    device const float* parameters [[buffer(7)]],
    constant uint& nEvents [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= nEvents ){ return; }

  float weight = baseWeights[gid];
  uint dialOffset = eventDialOffsets[gid];
  uint dialCount = eventDialCounts[gid];
  for( uint iDial = 0 ; iDial < dialCount ; iDial++ ){
    uint flatDial = dialOffset + iDial;
    float response = parameters[normParameterIndices[flatDial]];
    response = max(response, normMinResponses[flatDial]);
    response = min(response, normMaxResponses[flatDial]);
    weight *= response;
  }
  eventWeights[gid] = weight;
}

kernel void fill_histograms(
    device const float* eventWeights [[buffer(0)]],
    device const int* globalBins [[buffer(1)]],
    device float* histSums [[buffer(2)]],
    device float* histSumSquares [[buffer(3)]],
    constant uint& nEvents [[buffer(4)]],
    constant uint& totalBins [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= totalBins ){ return; }

  float sum = 0.0f;
  float sumCompensation = 0.0f;
  float sumSq = 0.0f;
  float sumSqCompensation = 0.0f;
  int bin = int(gid);
  for( uint iEvent = 0 ; iEvent < nEvents ; iEvent++ ){
    if( globalBins[iEvent] != bin ){ continue; }
    float weight = eventWeights[iEvent];
    float correctedWeight = weight - sumCompensation;
    float nextSum = sum + correctedWeight;
    sumCompensation = (nextSum - sum) - correctedWeight;
    sum = nextSum;

    float weightSq = weight * weight;
    float correctedWeightSq = weightSq - sumSqCompensation;
    float nextSumSq = sumSq + correctedWeightSq;
    sumSqCompensation = (nextSumSq - sumSq) - correctedWeightSq;
    sumSq = nextSumSq;
  }

  histSums[gid] = sum;
  histSumSquares[gid] = sumSq;
}

kernel void fill_histogram_partials_by_bin(
    device const float* eventWeights [[buffer(0)]],
    device const uint* binEventOffsets [[buffer(1)]],
    device const uint* binEventIndices [[buffer(2)]],
    device float* partialSums [[buffer(3)]],
    device float* partialSumSquares [[buffer(4)]],
    constant uint& totalBins [[buffer(5)]],
    constant uint& maxChunksPerBin [[buffer(6)]],
    constant uint& chunkSize [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  uint totalPartials = totalBins * maxChunksPerBin;
  if( gid >= totalPartials ){ return; }

  uint bin = gid / maxChunksPerBin;
  uint chunk = gid - bin * maxChunksPerBin;
  uint binBegin = binEventOffsets[bin];
  uint binEnd = binEventOffsets[bin + 1];
  uint begin = binBegin + chunk * chunkSize;
  uint end = min(begin + chunkSize, binEnd);

  float sum = 0.0f;
  float sumCompensation = 0.0f;
  float sumSq = 0.0f;
  float sumSqCompensation = 0.0f;
  for( uint iEntry = begin ; iEntry < end ; iEntry++ ){
    uint iEvent = binEventIndices[iEntry];
    float weight = eventWeights[iEvent];
    float correctedWeight = weight - sumCompensation;
    float nextSum = sum + correctedWeight;
    sumCompensation = (nextSum - sum) - correctedWeight;
    sum = nextSum;

    float weightSq = weight * weight;
    float correctedWeightSq = weightSq - sumSqCompensation;
    float nextSumSq = sumSq + correctedWeightSq;
    sumSqCompensation = (nextSumSq - sumSq) - correctedWeightSq;
    sumSq = nextSumSq;
  }

  partialSums[gid] = sum;
  partialSumSquares[gid] = sumSq;
}

kernel void finalize_histograms_from_partials(
    device const float* partialSums [[buffer(0)]],
    device const float* partialSumSquares [[buffer(1)]],
    device float* histSums [[buffer(2)]],
    device float* histSumSquares [[buffer(3)]],
    constant uint& totalBins [[buffer(4)]],
    constant uint& maxChunksPerBin [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= totalBins ){ return; }

  float sum = 0.0f;
  float sumCompensation = 0.0f;
  float sumSq = 0.0f;
  float sumSqCompensation = 0.0f;
  uint offset = gid * maxChunksPerBin;
  for( uint iChunk = 0 ; iChunk < maxChunksPerBin ; iChunk++ ){
    float partial = partialSums[offset + iChunk];
    float correctedPartial = partial - sumCompensation;
    float nextSum = sum + correctedPartial;
    sumCompensation = (nextSum - sum) - correctedPartial;
    sum = nextSum;

    float partialSq = partialSumSquares[offset + iChunk];
    float correctedPartialSq = partialSq - sumSqCompensation;
    float nextSumSq = sumSq + correctedPartialSq;
    sumSqCompensation = (nextSumSq - sumSq) - correctedPartialSq;
    sumSq = nextSumSq;
  }

  histSums[gid] = sum;
  histSumSquares[gid] = sumSq;
}
)METAL";

  template<typename T>
  id<MTLBuffer> makeBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    if( values.empty() ){ return nil; }
    return [device newBufferWithBytes:values.data()
                               length:values.size() * sizeof(T)
                              options:MTLResourceStorageModeShared];
  }

  id<MTLBuffer> makeEmptyBuffer(id<MTLDevice> device, std::size_t byteSize) {
    if( byteSize == 0 ){ return nil; }
    return [device newBufferWithLength:byteSize options:MTLResourceStorageModeShared];
  }

  void releaseBuffer(id<MTLBuffer>& buffer) {
    [buffer release];
    buffer = nil;
  }

  template<typename T>
  void copyToBuffer(id<MTLBuffer> buffer, const std::vector<T>& values) {
    if( buffer == nil or values.empty() ){ return; }
    std::memcpy(buffer.contents, values.data(), values.size() * sizeof(T));
  }

  id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,
                                           id<MTLLibrary> library,
                                           NSString* functionName) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    if( function == nil ){
      LogError << "Could not find " << [functionName UTF8String] << " Metal function." << std::endl;
      return nil;
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if( pipeline == nil and error != nil ){
      LogError << "Could not create MPS backend pipeline " << [functionName UTF8String]
               << ": " << [[error localizedDescription] UTF8String] << std::endl;
    }
    return pipeline;
  }
}

struct Backends::MpsBackend::Impl {
  struct Result {
    PropagationToken token{};
    PropagationStatus status{};
    std::vector<double> eventWeights{};
    std::vector<double> histSums{};
    std::vector<double> histSumSquares{};
    double likelihood{0};
  };

  id<MTLDevice> device{nil};
  id<MTLCommandQueue> commandQueue{nil};
  id<MTLComputePipelineState> normWeightsPipeline{nil};
  id<MTLComputePipelineState> histogramPipeline{nil};
  id<MTLComputePipelineState> histogramPartialsPipeline{nil};
  id<MTLComputePipelineState> histogramFinalizePipeline{nil};
  bool isAvailable{false};

  bool isDeviceModelSupported{false};
  static constexpr uint32_t histogramChunkSize{256};
  uint32_t maxHistogramChunksPerBin{1};
  id<MTLBuffer> eventWeightsBuffer{nil};
  id<MTLBuffer> baseWeightsBuffer{nil};
  id<MTLBuffer> eventDialOffsetsBuffer{nil};
  id<MTLBuffer> eventDialCountsBuffer{nil};
  id<MTLBuffer> globalBinsBuffer{nil};
  id<MTLBuffer> binEventOffsetsBuffer{nil};
  id<MTLBuffer> binEventIndicesBuffer{nil};
  id<MTLBuffer> normParameterIndicesBuffer{nil};
  id<MTLBuffer> normMinResponsesBuffer{nil};
  id<MTLBuffer> normMaxResponsesBuffer{nil};
  id<MTLBuffer> parametersBuffer{nil};
  id<MTLBuffer> partialHistSumsBuffer{nil};
  id<MTLBuffer> partialHistSumSquaresBuffer{nil};
  id<MTLBuffer> histSumsBuffer{nil};
  id<MTLBuffer> histSumSquaresBuffer{nil};
  id<MTLBuffer> nEventsBuffer{nil};
  id<MTLBuffer> totalBinsBuffer{nil};
  id<MTLBuffer> maxHistogramChunksPerBinBuffer{nil};
  id<MTLBuffer> histogramChunkSizeBuffer{nil};

  BackendModel model{};
  BackendLikelihoodModel likelihoodModel{};
  Result lastResult{};
  std::uint64_t nextTokenId{1};
  bool isBuilt{false};

  Impl() {
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

    normWeightsPipeline = makePipeline(device, library, @"compute_norm_event_weights");
    histogramPipeline = makePipeline(device, library, @"fill_histograms");
    histogramPartialsPipeline = makePipeline(device, library, @"fill_histogram_partials_by_bin");
    histogramFinalizePipeline = makePipeline(device, library, @"finalize_histograms_from_partials");
    [library release];
    if( normWeightsPipeline == nil or histogramPipeline == nil
        or histogramPartialsPipeline == nil or histogramFinalizePipeline == nil ){ return; }

    commandQueue = [device newCommandQueue];
    if( commandQueue == nil ){ return; }

    isAvailable = true;
  }

  ~Impl() {
    releaseDeviceBuffers();
    [normWeightsPipeline release];
    [histogramPipeline release];
    [histogramPartialsPipeline release];
    [histogramFinalizePipeline release];
    [commandQueue release];
    [device release];
  }

  void releaseDeviceBuffers() {
    releaseBuffer(eventWeightsBuffer);
    releaseBuffer(baseWeightsBuffer);
    releaseBuffer(eventDialOffsetsBuffer);
    releaseBuffer(eventDialCountsBuffer);
    releaseBuffer(globalBinsBuffer);
    releaseBuffer(binEventOffsetsBuffer);
    releaseBuffer(binEventIndicesBuffer);
    releaseBuffer(normParameterIndicesBuffer);
    releaseBuffer(normMinResponsesBuffer);
    releaseBuffer(normMaxResponsesBuffer);
    releaseBuffer(parametersBuffer);
    releaseBuffer(partialHistSumsBuffer);
    releaseBuffer(partialHistSumSquaresBuffer);
    releaseBuffer(histSumsBuffer);
    releaseBuffer(histSumSquaresBuffer);
    releaseBuffer(nEventsBuffer);
    releaseBuffer(totalBinsBuffer);
    releaseBuffer(maxHistogramChunksPerBinBuffer);
    releaseBuffer(histogramChunkSizeBuffer);
  }

  bool isCurrentToken(const PropagationToken& token_) const {
    return token_.isValid and lastResult.token.isValid and token_.id == lastResult.token.id;
  }

  void resetResult(const PropagationRequest& request_) {
    lastResult.token.id = nextTokenId++;
    lastResult.token.isValid = true;
    lastResult.status = PropagationStatus();
    lastResult.status.backend = BackendStatus::Running;
    lastResult.eventWeights.clear();
    lastResult.histSums.clear();
    lastResult.histSumSquares.clear();
    lastResult.likelihood = 0;

    for( auto request : request_.outputs ){
      lastResult.status.state(request) = OutputState::Scheduled;
    }
  }

  void applyParameterSnapshot(const ParameterSnapshot& parameters_) {
    if( parameters_.empty() ){ return; }
    for( std::size_t iPar = 0 ; iPar < parameters_.values.size() ; iPar++ ){
      auto* parPtr = const_cast<Parameter*>(model.parameters.at(iPar));
      parPtr->setParameterValue(parameters_.values.at(iPar), true);
    }
  }

  void updateInputBuffers() {
    for( const auto* inputBuffer : model.inputBuffers ){
      const_cast<DialInputBuffer*>(inputBuffer)->update();
    }
  }

  int findParameterIndex(const Parameter* parameter_) const {
    for( std::size_t iPar = 0 ; iPar < model.parameters.size() ; iPar++ ){
      if( model.parameters[iPar] == parameter_ ){ return int(iPar); }
    }
    return -1;
  }

  bool buildDeviceModel() {
    releaseDeviceBuffers();
    isDeviceModelSupported = false;

    if( not isAvailable ){ return false; }
    if( model.events.empty() or model.totalBins <= 0 ){ return false; }

    std::vector<float> baseWeights(model.events.size());
    std::vector<uint32_t> eventDialOffsets(model.events.size());
    std::vector<uint32_t> eventDialCounts(model.events.size());
    std::vector<int> globalBins(model.events.size());
    std::vector<uint32_t> eventsPerBin(model.totalBins, 0);
    std::vector<uint32_t> normParameterIndices(model.eventDials.size());
    std::vector<float> normMinResponses(model.eventDials.size(), -std::numeric_limits<float>::infinity());
    std::vector<float> normMaxResponses(model.eventDials.size(), std::numeric_limits<float>::infinity());

    for( const auto& event : model.events ){
      if( event.globalBinIndex < 0 or event.globalBinIndex >= model.totalBins ){ return false; }
      baseWeights[event.resultIndex] = float(event.baseWeight);
      eventDialOffsets[event.resultIndex] = uint32_t(event.firstDial);
      eventDialCounts[event.resultIndex] = uint32_t(event.dialCount);
      globalBins[event.resultIndex] = event.globalBinIndex;
      eventsPerBin[event.globalBinIndex]++;
    }

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

    for( std::size_t iDial = 0 ; iDial < model.eventDials.size() ; iDial++ ){
      const auto* interface = model.eventDials[iDial].interface;
      if( interface == nullptr or dynamic_cast<const Norm*>(interface->getDialBaseRef()) == nullptr ){
        return false;
      }

      const auto* inputBuffer = interface->getInputBufferRef();
      if( inputBuffer == nullptr or inputBuffer->getBufferSize() != 1 ){
        return false;
      }

      int parameterIndex = findParameterIndex(&inputBuffer->getParameter(0));
      if( parameterIndex < 0 ){ return false; }
      normParameterIndices[iDial] = uint32_t(parameterIndex);

      const auto* supervisor = interface->getResponseSupervisorRef();
      if( supervisor != nullptr ){
        if( not std::isnan(supervisor->getMinResponse()) ){
          normMinResponses[iDial] = float(supervisor->getMinResponse());
        }
        if( not std::isnan(supervisor->getMaxResponse()) ){
          normMaxResponses[iDial] = float(supervisor->getMaxResponse());
        }
      }
    }

    uint32_t nEvents = uint32_t(model.events.size());
    uint32_t totalBins = uint32_t(model.totalBins);
    uint32_t chunkSize = histogramChunkSize;
    uint32_t totalPartials = totalBins * maxHistogramChunksPerBin;

    baseWeightsBuffer = makeBuffer(device, baseWeights);
    eventDialOffsetsBuffer = makeBuffer(device, eventDialOffsets);
    eventDialCountsBuffer = makeBuffer(device, eventDialCounts);
    globalBinsBuffer = makeBuffer(device, globalBins);
    binEventOffsetsBuffer = makeBuffer(device, binEventOffsets);
    binEventIndicesBuffer = makeBuffer(device, binEventIndices);
    normParameterIndicesBuffer = makeBuffer(device, normParameterIndices);
    normMinResponsesBuffer = makeBuffer(device, normMinResponses);
    normMaxResponsesBuffer = makeBuffer(device, normMaxResponses);
    eventWeightsBuffer = makeEmptyBuffer(device, model.events.size() * sizeof(float));
    parametersBuffer = makeEmptyBuffer(device, model.parameters.size() * sizeof(float));
    partialHistSumsBuffer = makeEmptyBuffer(device, std::size_t(totalPartials) * sizeof(float));
    partialHistSumSquaresBuffer = makeEmptyBuffer(device, std::size_t(totalPartials) * sizeof(float));
    histSumsBuffer = makeEmptyBuffer(device, std::size_t(model.totalBins) * sizeof(float));
    histSumSquaresBuffer = makeEmptyBuffer(device, std::size_t(model.totalBins) * sizeof(float));
    nEventsBuffer = [device newBufferWithBytes:&nEvents length:sizeof(nEvents) options:MTLResourceStorageModeShared];
    totalBinsBuffer = [device newBufferWithBytes:&totalBins length:sizeof(totalBins) options:MTLResourceStorageModeShared];
    maxHistogramChunksPerBinBuffer = [device newBufferWithBytes:&maxHistogramChunksPerBin
                                                         length:sizeof(maxHistogramChunksPerBin)
                                                        options:MTLResourceStorageModeShared];
    histogramChunkSizeBuffer = [device newBufferWithBytes:&chunkSize
                                                   length:sizeof(chunkSize)
                                                  options:MTLResourceStorageModeShared];

    if( baseWeightsBuffer == nil or eventDialOffsetsBuffer == nil or eventDialCountsBuffer == nil
        or globalBinsBuffer == nil or binEventOffsetsBuffer == nil or binEventIndicesBuffer == nil
        or normParameterIndicesBuffer == nil or normMinResponsesBuffer == nil
        or normMaxResponsesBuffer == nil or eventWeightsBuffer == nil or parametersBuffer == nil
        or partialHistSumsBuffer == nil or partialHistSumSquaresBuffer == nil
        or histSumsBuffer == nil or histSumSquaresBuffer == nil or nEventsBuffer == nil
        or totalBinsBuffer == nil or maxHistogramChunksPerBinBuffer == nil
        or histogramChunkSizeBuffer == nil ){
      releaseDeviceBuffers();
      return false;
    }

    isDeviceModelSupported = true;
    return true;
  }

  void updateDeviceParameters() {
    std::vector<float> parameterValues(model.parameters.size());
    for( std::size_t iPar = 0 ; iPar < model.parameters.size() ; iPar++ ){
      parameterValues[iPar] = float(model.parameters[iPar]->getParameterValue());
    }
    copyToBuffer(parametersBuffer, parameterValues);
  }

  bool encodeNormEventWeights(id<MTLComputeCommandEncoder> encoder) {
    if( not isDeviceModelSupported ){ return false; }
    [encoder setComputePipelineState:normWeightsPipeline];
    [encoder setBuffer:eventWeightsBuffer offset:0 atIndex:0];
    [encoder setBuffer:baseWeightsBuffer offset:0 atIndex:1];
    [encoder setBuffer:eventDialOffsetsBuffer offset:0 atIndex:2];
    [encoder setBuffer:eventDialCountsBuffer offset:0 atIndex:3];
    [encoder setBuffer:normParameterIndicesBuffer offset:0 atIndex:4];
    [encoder setBuffer:normMinResponsesBuffer offset:0 atIndex:5];
    [encoder setBuffer:normMaxResponsesBuffer offset:0 atIndex:6];
    [encoder setBuffer:parametersBuffer offset:0 atIndex:7];
    [encoder setBuffer:nEventsBuffer offset:0 atIndex:8];

    NSUInteger width = std::min<NSUInteger>(normWeightsPipeline.maxTotalThreadsPerThreadgroup, 256);
    if( width == 0 ){ width = 1; }
    [encoder dispatchThreads:MTLSizeMake(NSUInteger(model.events.size()), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    return true;
  }

  bool encodeHistogramsFromDeviceWeights(id<MTLComputeCommandEncoder> encoder) {
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

  bool runDevicePropagation(bool needHistograms_) {
    if( not isDeviceModelSupported ){ return false; }
    updateDeviceParameters();

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if( not encodeNormEventWeights(encoder) ){
      [encoder endEncoding];
      return false;
    }
    if( needHistograms_ and not encodeHistogramsFromDeviceWeights(encoder) ){
      [encoder endEncoding];
      return false;
    }
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    if( commandBuffer.status != MTLCommandBufferStatusCompleted ){
      return false;
    }

    if( needHistograms_ ){
      auto* histSums = static_cast<float*>(histSumsBuffer.contents);
      auto* histSumSquares = static_cast<float*>(histSumSquaresBuffer.contents);
      lastResult.histSums.resize(model.totalBins);
      lastResult.histSumSquares.resize(model.totalBins);
      for( int iBin = 0 ; iBin < model.totalBins ; iBin++ ){
        lastResult.histSums[iBin] = histSums[iBin];
        lastResult.histSumSquares[iBin] = histSumSquares[iBin];
      }
    }

    return true;
  }

  void copyDeviceEventWeightsToHostResult() {
    LogThrowIf(eventWeightsBuffer == nil);
    auto* weights = static_cast<float*>(eventWeightsBuffer.contents);
    lastResult.eventWeights.resize(model.events.size());
    for( const auto& event : model.events ){
      lastResult.eventWeights[event.resultIndex] = weights[event.resultIndex];
    }
  }

  void calculateEventWeights() {
    lastResult.eventWeights.resize(model.events.size());

    for( const auto& event : model.events ){
      double weight = event.baseWeight;

      for( std::size_t iDial = 0 ; iDial < event.dialCount ; iDial++ ){
        const auto& dialRef = model.eventDials[event.firstDial + iDial];
        weight *= dialRef.interface->evalResponse();
      }

      lastResult.eventWeights[event.resultIndex] = weight;
    }
  }

  bool calculateHistogramsOnDevice() {
    if( not isAvailable ){ return false; }
    if( model.totalBins <= 0 ){ return false; }
    if( model.events.empty() ){
      lastResult.histSums.assign(model.totalBins, 0);
      lastResult.histSumSquares.assign(model.totalBins, 0);
      return true;
    }

    if( lastResult.eventWeights.size() != model.events.size() ){
      calculateEventWeights();
    }

    std::vector<float> eventWeightsFloat(lastResult.eventWeights.size());
    std::vector<int> globalBins(model.events.size());
    for( const auto& event : model.events ){
      eventWeightsFloat[event.resultIndex] = float(lastResult.eventWeights[event.resultIndex]);
      globalBins[event.resultIndex] = event.globalBinIndex;
    }

    auto eventWeightsBuffer = makeBuffer(device, eventWeightsFloat);
    auto globalBinsBuffer = makeBuffer(device, globalBins);
    auto histSumsBuffer = [device newBufferWithLength:std::size_t(model.totalBins) * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    auto histSumSquaresBuffer = [device newBufferWithLength:std::size_t(model.totalBins) * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    uint32_t nEvents = uint32_t(model.events.size());
    uint32_t totalBins = uint32_t(model.totalBins);
    auto nEventsBuffer = [device newBufferWithBytes:&nEvents length:sizeof(nEvents) options:MTLResourceStorageModeShared];
    auto totalBinsBuffer = [device newBufferWithBytes:&totalBins length:sizeof(totalBins) options:MTLResourceStorageModeShared];

    if( eventWeightsBuffer == nil or globalBinsBuffer == nil or histSumsBuffer == nil
        or histSumSquaresBuffer == nil or nEventsBuffer == nil or totalBinsBuffer == nil ){
      [eventWeightsBuffer release];
      [globalBinsBuffer release];
      [histSumsBuffer release];
      [histSumSquaresBuffer release];
      [nEventsBuffer release];
      [totalBinsBuffer release];
      return false;
    }

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:histogramPipeline];
    [encoder setBuffer:eventWeightsBuffer offset:0 atIndex:0];
    [encoder setBuffer:globalBinsBuffer offset:0 atIndex:1];
    [encoder setBuffer:histSumsBuffer offset:0 atIndex:2];
    [encoder setBuffer:histSumSquaresBuffer offset:0 atIndex:3];
    [encoder setBuffer:nEventsBuffer offset:0 atIndex:4];
    [encoder setBuffer:totalBinsBuffer offset:0 atIndex:5];

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
      auto* histSums = static_cast<float*>(histSumsBuffer.contents);
      auto* histSumSquares = static_cast<float*>(histSumSquaresBuffer.contents);
      lastResult.histSums.resize(model.totalBins);
      lastResult.histSumSquares.resize(model.totalBins);
      for( int iBin = 0 ; iBin < model.totalBins ; iBin++ ){
        lastResult.histSums[iBin] = histSums[iBin];
        lastResult.histSumSquares[iBin] = histSumSquares[iBin];
      }
    }

    [eventWeightsBuffer release];
    [globalBinsBuffer release];
    [histSumsBuffer release];
    [histSumSquaresBuffer release];
    [nEventsBuffer release];
    [totalBinsBuffer release];
    return ok;
  }

  void calculateLikelihood() {
    lastResult.likelihood = 0;
    for( const auto& sample : likelihoodModel.samples ){
      for( int iBin = 0 ; iBin < int(sample.dataSums.size()) ; iBin++ ){
        if( iBin < int(sample.ignoredBins.size()) and sample.ignoredBins[iBin] ){ continue; }
        int globalBin = sample.binOffset + iBin;
        double pred = lastResult.histSums[globalBin];
        double predErr = std::sqrt(lastResult.histSumSquares[globalBin]);
        lastResult.likelihood += sample.evalBin(sample.dataSums[iBin], pred, predErr, iBin);
      }
    }
  }

  void materializeEventWeights() {
    if( lastResult.eventWeights.empty() and eventWeightsBuffer != nil ){
      copyDeviceEventWeightsToHostResult();
    }
    LogThrowIf(lastResult.eventWeights.size() != model.events.size());
    for( const auto& event : model.events ){
      event.event->getWeights().current = lastResult.eventWeights[event.resultIndex];
    }
  }

  void materializeHistograms() {
    LogThrowIf(lastResult.histSums.size() != std::size_t(model.totalBins));
    LogThrowIf(lastResult.histSumSquares.size() != std::size_t(model.totalBins));

    for( const auto& sample : model.samples ){
      auto& binContentList = sample.histogram->getBinContentList();
      auto& binContextList = sample.histogram->getBinContextList();

      for( auto& binContent : binContentList ){
        binContent.sumWeights = 0;
        binContent.sqrtSumSqWeights = 0;
      }

      for( auto& binContext : binContextList ){
        int globalBin = sample.binOffset + binContext.bin.getIndex();
        auto& binContent = binContentList[binContext.bin.getIndex()];
        binContent.sumWeights = lastResult.histSums[globalBin];
        binContent.sqrtSumSqWeights = std::sqrt(lastResult.histSumSquares[globalBin]);
      }
    }
  }
};

Backends::MpsBackend::MpsBackend() : _impl_(std::make_unique<Impl>()) {}
Backends::MpsBackend::~MpsBackend() = default;

Backends::BackendCapabilities Backends::MpsBackend::getCapabilities() const {
  BackendCapabilities out;
  out.supportsGpu = true;
  out.supportsEventWeights = true;
  out.supportsHistograms = true;
  out.supportsLikelihood = true;
  out.deviceName = _impl_->isAvailable ? [[_impl_->device name] UTF8String] : "Metal unavailable";
  return out;
}

void Backends::MpsBackend::build(const BackendModel& model_) {
  _impl_->model = model_;
  _impl_->lastResult = Impl::Result();
  _impl_->buildDeviceModel();
  _impl_->isBuilt = true;
}

void Backends::MpsBackend::setLikelihoodModel(const BackendLikelihoodModel& likelihoodModel_) {
  _impl_->likelihoodModel = likelihoodModel_;
}

Backends::PropagationToken Backends::MpsBackend::requestPropagation(
    const ParameterSnapshot& parameters_,
    const PropagationRequest& request_) {
  LogThrowIf(not _impl_->isBuilt, "MpsBackend has not been built.");
  LogThrowIf(not parameters_.empty() and parameters_.values.size() != _impl_->model.parameters.size(),
             "ParameterSnapshot size mismatch: " << parameters_.values.size()
                                                 << " != " << _impl_->model.parameters.size());

  _impl_->resetResult(request_);

  if( not _impl_->isAvailable ){
    _impl_->lastResult.status.backend = BackendStatus::Unavailable;
    for( auto request : request_.outputs ){
      _impl_->lastResult.status.state(request) = OutputState::Failed;
    }
    _impl_->lastResult.token.isValid = false;
    return {};
  }

  _impl_->applyParameterSnapshot(parameters_);
  _impl_->updateInputBuffers();

  bool needsEventWeights = request_.has(OutputRequest::EventWeights);
  bool needsHistograms = request_.has(OutputRequest::Histograms) or request_.has(OutputRequest::Likelihood);
  bool usedDevicePropagation = false;

  if( needsEventWeights or needsHistograms ){
    usedDevicePropagation = _impl_->runDevicePropagation(needsHistograms);
    if( usedDevicePropagation ){
      if( needsEventWeights ){
        _impl_->lastResult.status.eventWeights = OutputState::ReadyOnDevice;
      }
      if( request_.has(OutputRequest::Histograms) ){
        _impl_->lastResult.status.histograms = OutputState::ReadyOnDevice;
      }
    }
  }

  if( request_.has(OutputRequest::EventWeights) and not usedDevicePropagation ){
    _impl_->calculateEventWeights();
    _impl_->lastResult.status.eventWeights = OutputState::ReadyOnHost;
  }

  if( needsHistograms and not usedDevicePropagation ){
    if( not _impl_->calculateHistogramsOnDevice() ){
      _impl_->lastResult.status.backend = BackendStatus::Failed;
      if( request_.has(OutputRequest::Histograms) ){
        _impl_->lastResult.status.histograms = OutputState::Failed;
      }
      if( request_.has(OutputRequest::Likelihood) ){
        _impl_->lastResult.status.likelihood = OutputState::Failed;
      }
      return _impl_->lastResult.token;
    }
    if( request_.has(OutputRequest::Histograms) ){
      _impl_->lastResult.status.histograms = OutputState::ReadyOnDevice;
    }
  }

  if( request_.has(OutputRequest::Likelihood) ){
    if( _impl_->likelihoodModel.empty() ){
      _impl_->lastResult.status.likelihood = OutputState::Failed;
    }
    else{
      _impl_->calculateLikelihood();
      _impl_->lastResult.status.likelihood = OutputState::ReadyOnHost;
    }
  }

  if( request_.has(OutputRequest::BinIndices) ){
    _impl_->lastResult.status.binIndices = OutputState::Failed;
  }
  if( request_.has(OutputRequest::ObservableValues) ){
    _impl_->lastResult.status.observableValues = OutputState::Failed;
  }

  _impl_->lastResult.status.backend = BackendStatus::Ready;
  return _impl_->lastResult.token;
}

Backends::PropagationStatus Backends::MpsBackend::getStatus(const PropagationToken& token_) const {
  if( not _impl_->isCurrentToken(token_) ){
    PropagationStatus out;
    out.backend = BackendStatus::Failed;
    return out;
  }
  return _impl_->lastResult.status;
}

bool Backends::MpsBackend::isReady(const PropagationToken& token_) const {
  return _impl_->isCurrentToken(token_) and _impl_->lastResult.status.backend == BackendStatus::Ready;
}

void Backends::MpsBackend::wait(const PropagationToken& token_) {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
}

void Backends::MpsBackend::materialize(const PropagationToken& token_, OutputRequest output_) {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
  LogThrowIf(_impl_->lastResult.status.state(output_) != OutputState::ReadyOnDevice
             and _impl_->lastResult.status.state(output_) != OutputState::ReadyOnHost,
             "Requested backend output is not ready.");

  if( output_ == OutputRequest::EventWeights ){
    _impl_->materializeEventWeights();
    _impl_->lastResult.status.eventWeights = OutputState::ReadyOnHost;
  }
  else if( output_ == OutputRequest::Histograms ){
    _impl_->materializeHistograms();
    _impl_->lastResult.status.histograms = OutputState::ReadyOnHost;
  }
  else if( output_ == OutputRequest::Likelihood ){
    _impl_->lastResult.status.likelihood = OutputState::ReadyOnHost;
  }
  else{
    LogThrow("MpsBackend cannot materialize requested output yet.");
  }
}

double Backends::MpsBackend::getLikelihood(const PropagationToken& token_) const {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
  LogThrowIf(_impl_->lastResult.status.likelihood != OutputState::ReadyOnDevice
             and _impl_->lastResult.status.likelihood != OutputState::ReadyOnHost,
             "Backend likelihood is not ready.");
  return _impl_->lastResult.likelihood;
}
