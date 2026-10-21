#ifndef GUNDAM_MPS_BACKEND_INTERNAL_H
#define GUNDAM_MPS_BACKEND_INTERNAL_H

#include "MpsBackend.h"

#include "EngineView.h"
#include "PropagationResult.h"
#include "MpsEventPropagation.h"
#include "MpsWeightTypes.h"
#include "BackendTypes.h"
#include "MpsBackendKernelSource.h"
#include "ParameterSnapshot.h"


#include "GundamGlobals.h"
#include "Logger.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Backends {

  struct MpsBackendImpl {
    using Result = PropagationResult;

    id<MTLDevice> device{nil};
    id<MTLCommandQueue> commandQueue{nil};
    id<MTLComputePipelineState> eventWeightsPipeline{nil};
    id<MTLComputePipelineState> reducedEventWeightsPipeline{nil};
    id<MTLComputePipelineState> cachedCompactResponsesPipeline{nil};
    id<MTLComputePipelineState> reducedCachedCompactResponsesPipeline{nil};
    id<MTLComputePipelineState> cachedUniformResponsesPipeline{nil};
    id<MTLComputePipelineState> reducedCachedUniformResponsesPipeline{nil};
    id<MTLComputePipelineState> cachedMonotonicResponsesPipeline{nil};
    id<MTLComputePipelineState> reducedCachedMonotonicResponsesPipeline{nil};
    id<MTLComputePipelineState> cachedGeneralResponsesPipeline{nil};
    id<MTLComputePipelineState> reducedCachedGeneralResponsesPipeline{nil};
    id<MTLComputePipelineState> cachedGraphResponsesPipeline{nil};
    id<MTLComputePipelineState> reducedCachedGraphResponsesPipeline{nil};
    id<MTLComputePipelineState> histogramPipeline{nil};
    id<MTLComputePipelineState> histogramPartialsPipeline{nil};
    id<MTLComputePipelineState> histogramFinalizePipeline{nil};
    std::unique_ptr<MpsEventPropagation> eventPropagation{};
    bool useEventPropagation{false};
    bool isAvailable{false};
    bool enableCachedDialResponses{true};
    // The full descriptor scan is retained only for controlled diagnostic A/B runs.
    bool useCompactCachedDialIndices{true};
    bool useReducedDialDescriptors{true};
    bool enableDiagnosticDescriptorComparison{false};
    bool enableDiagnosticStageProfiling{false};
    std::map<std::string, double> diagnosticStageGpuSeconds{};

    bool isWeightModelSupported{false};
    static constexpr uint32_t histogramChunkSize{256};
    uint32_t maxHistogramChunksPerBin{1};
    id<MTLBuffer> eventWeightsBuffer{nil};
    id<MTLBuffer> baseWeightsBuffer{nil};
    id<MTLBuffer> eventDialRangesBuffer{nil};
    id<MTLBuffer> normDialOccurrencesBuffer{nil};
    id<MTLBuffer> reducedNormDialOccurrencesBuffer{nil};
    id<MTLBuffer> compactDialIndicesBuffer{nil};
    id<MTLBuffer> uniformDialIndicesBuffer{nil};
    id<MTLBuffer> monotonicDialIndicesBuffer{nil};
    id<MTLBuffer> generalDialIndicesBuffer{nil};
    id<MTLBuffer> graphDialIndicesBuffer{nil};
    id<MTLBuffer> compactDialDescriptorsBuffer{nil};
    id<MTLBuffer> reducedCompactDialDescriptorsBuffer{nil};
    id<MTLBuffer> uniformDialDescriptorsBuffer{nil};
    id<MTLBuffer> reducedUniformDialDescriptorsBuffer{nil};
    id<MTLBuffer> monotonicDialDescriptorsBuffer{nil};
    id<MTLBuffer> reducedMonotonicDialDescriptorsBuffer{nil};
    id<MTLBuffer> generalDialDescriptorsBuffer{nil};
    id<MTLBuffer> reducedGeneralDialDescriptorsBuffer{nil};
    id<MTLBuffer> graphDialDescriptorsBuffer{nil};
    id<MTLBuffer> reducedGraphDialDescriptorsBuffer{nil};
    id<MTLBuffer> compactCachedResponsesBuffer{nil};
    id<MTLBuffer> compactCachedDialIndicesBuffer{nil};
    id<MTLBuffer> uniformCachedResponsesBuffer{nil};
    id<MTLBuffer> uniformCachedDialIndicesBuffer{nil};
    id<MTLBuffer> monotonicCachedResponsesBuffer{nil};
    id<MTLBuffer> monotonicCachedDialIndicesBuffer{nil};
    id<MTLBuffer> generalCachedResponsesBuffer{nil};
    id<MTLBuffer> generalCachedDialIndicesBuffer{nil};
    id<MTLBuffer> graphCachedResponsesBuffer{nil};
    id<MTLBuffer> graphCachedDialIndicesBuffer{nil};
    id<MTLBuffer> globalBinsBuffer{nil};
    id<MTLBuffer> binEventOffsetsBuffer{nil};
    id<MTLBuffer> binEventIndicesBuffer{nil};
    id<MTLBuffer> splineDataBuffer{nil};
    id<MTLBuffer> responseSmoothingBuffer{nil};
    id<MTLBuffer> parametersBuffer{nil};
    id<MTLBuffer> externalWeightsBuffer{nil};
    id<MTLBuffer> externalDialOccurrencesBuffer{nil};
    id<MTLBuffer> reducedExternalDialOccurrencesBuffer{nil};
    std::vector<std::uint64_t> externalWeightGenerations{};
    id<MTLBuffer> partialHistSumsBuffer{nil};
    id<MTLBuffer> partialHistSumSquaresBuffer{nil};
    id<MTLBuffer> histSumsBuffer{nil};
    id<MTLBuffer> histSumSquaresBuffer{nil};
    id<MTLBuffer> eventWeightsReadbackBuffer{nil};
    id<MTLBuffer> histSumsReadbackBuffer{nil};
    id<MTLBuffer> histSumSquaresReadbackBuffer{nil};
    id<MTLBuffer> nEventsBuffer{nil};
    id<MTLBuffer> totalBinsBuffer{nil};
    id<MTLBuffer> maxHistogramChunksPerBinBuffer{nil};
    id<MTLBuffer> histogramChunkSizeBuffer{nil};

    EngineView engineView{};
    PropagationView& model;
    LikelihoodView& likelihoodModel;
    Result lastResult{};
    BackendTimingSummary buildTiming{};
    BackendTimingSummary lastTiming{};
    std::string deviceModelFallbackReason{};
    std::uint64_t nextTokenId{1};
    bool isBuilt{false};
    uint32_t cachedDialCount{0};
    uint32_t compactCachedDialCount{0};
    uint32_t uniformCachedDialCount{0};
    uint32_t monotonicCachedDialCount{0};
    uint32_t generalCachedDialCount{0};
    uint32_t graphCachedDialCount{0};
    uint32_t compactDialDescriptorCount{0};
    uint32_t uniformDialDescriptorCount{0};
    uint32_t monotonicDialDescriptorCount{0};
    uint32_t generalDialDescriptorCount{0};
    uint32_t graphDialDescriptorCount{0};

    MpsBackendImpl();
    ~MpsBackendImpl();

    void releaseDeviceBuffers();
    [[nodiscard]] bool isCurrentToken(const PropagationToken& token_) const;
    void resetResult();

    [[nodiscard]] bool buildWeightModel();
    void updateDeviceParameters(const ParameterSnapshot& parameters_);
    void updateExternalWeights(const PropagationInputs& inputs_);
    [[nodiscard]] bool encodeEventWeights(id<MTLComputeCommandEncoder> encoder);
    [[nodiscard]] bool encodeCachedDialResponses(id<MTLComputeCommandEncoder> encoder,
                                                 id<MTLComputePipelineState> pipeline_,
                                                 id<MTLBuffer> cachedResponsesBuffer_,
                                                 id<MTLBuffer> descriptorsBuffer_,
                                                 uint32_t descriptorCount_,
                                                 id<MTLBuffer> cachedDialIndicesBuffer_,
                                                 uint32_t cachedDialCount_);
    [[nodiscard]] bool encodeHistogramsFromDeviceWeights(id<MTLComputeCommandEncoder> encoder);
    [[nodiscard]] bool runWeightPropagation(const PropagationInputs& inputs_, bool needHistograms_);
    void copyDeviceEventWeightsToHostResult();
    [[nodiscard]] bool calculateHistogramsOnDevice();
    void calculateLikelihood();
    void materializeEventWeights();
    void materializeHistograms();
  };

}

namespace {
  static NSString* const kMpsBackendMetalSource = GUNDAM_MPS_BACKEND_KERNEL_SOURCE;

  template<typename T>
  id<MTLBuffer> makeSharedBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    if( values.empty() ){ return nil; }
    return [device newBufferWithBytes:values.data()
                               length:values.size() * sizeof(T)
                              options:MTLResourceStorageModeShared];
  }

  inline id<MTLBuffer> makeSharedEmptyBuffer(id<MTLDevice> device, std::size_t byteSize) {
    if( byteSize == 0 ){ return nil; }
    return [device newBufferWithLength:byteSize options:MTLResourceStorageModeShared];
  }

  inline id<MTLBuffer> makePrivateEmptyBuffer(id<MTLDevice> device, std::size_t byteSize) {
    if( byteSize == 0 ){ return nil; }
    return [device newBufferWithLength:byteSize options:MTLResourceStorageModePrivate];
  }

  template<typename T>
  id<MTLBuffer> makePrivateBuffer(id<MTLDevice> device,
                                  id<MTLCommandQueue> commandQueue,
                                  const std::vector<T>& values) {
    if( values.empty() ){ return nil; }
    std::size_t byteSize = values.size() * sizeof(T);
    id<MTLBuffer> privateBuffer = makePrivateEmptyBuffer(device, byteSize);
    id<MTLBuffer> stagingBuffer = [device newBufferWithBytes:values.data()
                                                      length:byteSize
                                                     options:MTLResourceStorageModeShared];
    if( privateBuffer == nil or stagingBuffer == nil ){
      [privateBuffer release];
      [stagingBuffer release];
      return nil;
    }

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
    [encoder copyFromBuffer:stagingBuffer
               sourceOffset:0
                   toBuffer:privateBuffer
          destinationOffset:0
                       size:byteSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    [stagingBuffer release];
    if( commandBuffer.status != MTLCommandBufferStatusCompleted ){
      [privateBuffer release];
      return nil;
    }
    return privateBuffer;
  }

  inline void releaseBuffer(id<MTLBuffer>& buffer) {
    [buffer release];
    buffer = nil;
  }

  [[nodiscard]] inline double secondsSince(std::chrono::steady_clock::time_point start_) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  }

  template<typename Scalar = float>
  inline void uploadParameters(id<MTLBuffer> destination_, const Backends::ParameterSnapshot& parameters_,
                               Backends::BackendTimingSummary& timing_) {
    auto start = std::chrono::steady_clock::now();
    auto* destination = static_cast<Scalar*>(destination_.contents);
    for( std::size_t i = 0; i < parameters_.values.size(); ++i ){ destination[i] = Scalar(parameters_.values[i]); }
    timing_.parameterUploadSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }

  template<typename Scalar = float>
  inline void uploadExternalWeights(id<MTLBuffer> destination_, std::vector<std::uint64_t>& generations_,
                                    const Backends::PropagationInputs& inputs_, Backends::BackendTimingSummary& timing_) {
    auto start = std::chrono::steady_clock::now();
    auto* destination = static_cast<Scalar*>(destination_.contents);
    for( std::size_t i = 0; i < inputs_.externalWeights.size(); ++i ){
      const auto& block = inputs_.externalWeights[i];
      if( generations_[i] == block.generation ){ continue; }
      if( block.count ){
        std::transform(block.values, block.values + block.count, destination + block.destinationOffset,
                       [](double value_){ return Scalar(value_); });
      }
      generations_[i] = block.generation;
      timing_.externalWeightUploadBytes += block.count * sizeof(Scalar);
      ++timing_.externalWeightUploadBlocks;
    }
    timing_.externalWeightUploadSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }

  [[nodiscard]] inline id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,
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

#endif // GUNDAM_MPS_BACKEND_INTERNAL_H
