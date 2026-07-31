#ifndef GUNDAM_MPS_BACKEND_INTERNAL_H
#define GUNDAM_MPS_BACKEND_INTERNAL_H

#include "MpsEngineBackend.h"

#include "EngineView.h"
#include "BackendTypes.h"
#include "MpsBackendKernelSource.h"
#include "ParameterSnapshot.h"

#include "CompactSpline.h"
#include "GeneralSpline.h"
#include "Graph.h"
#include "Norm.h"
#include "MonotonicSpline.h"
#include "Shift.h"
#include "UniformSpline.h"

#include "GundamGlobals.h"
#include "Logger.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Backends {

  struct MpsBackendImpl {
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
    id<MTLComputePipelineState> eventWeightsPipeline{nil};
    id<MTLComputePipelineState> cachedCompactResponsesPipeline{nil};
    id<MTLComputePipelineState> cachedUniformResponsesPipeline{nil};
    id<MTLComputePipelineState> cachedMonotonicResponsesPipeline{nil};
    id<MTLComputePipelineState> cachedGeneralResponsesPipeline{nil};
    id<MTLComputePipelineState> cachedGraphResponsesPipeline{nil};
    id<MTLComputePipelineState> histogramPipeline{nil};
    id<MTLComputePipelineState> histogramPartialsPipeline{nil};
    id<MTLComputePipelineState> histogramFinalizePipeline{nil};
    bool isAvailable{false};

    bool isDeviceModelSupported{false};
    static constexpr uint32_t histogramChunkSize{256};
    uint32_t maxHistogramChunksPerBin{1};
    id<MTLBuffer> eventWeightsBuffer{nil};
    id<MTLBuffer> baseWeightsBuffer{nil};
    id<MTLBuffer> eventDialRangesBuffer{nil};
    id<MTLBuffer> normDialOccurrencesBuffer{nil};
    id<MTLBuffer> compactDialIndicesBuffer{nil};
    id<MTLBuffer> uniformDialIndicesBuffer{nil};
    id<MTLBuffer> monotonicDialIndicesBuffer{nil};
    id<MTLBuffer> generalDialIndicesBuffer{nil};
    id<MTLBuffer> graphDialIndicesBuffer{nil};
    id<MTLBuffer> compactDialDescriptorsBuffer{nil};
    id<MTLBuffer> uniformDialDescriptorsBuffer{nil};
    id<MTLBuffer> monotonicDialDescriptorsBuffer{nil};
    id<MTLBuffer> generalDialDescriptorsBuffer{nil};
    id<MTLBuffer> graphDialDescriptorsBuffer{nil};
    id<MTLBuffer> compactCachedResponsesBuffer{nil};
    id<MTLBuffer> uniformCachedResponsesBuffer{nil};
    id<MTLBuffer> monotonicCachedResponsesBuffer{nil};
    id<MTLBuffer> generalCachedResponsesBuffer{nil};
    id<MTLBuffer> graphCachedResponsesBuffer{nil};
    id<MTLBuffer> globalBinsBuffer{nil};
    id<MTLBuffer> binEventOffsetsBuffer{nil};
    id<MTLBuffer> binEventIndicesBuffer{nil};
    id<MTLBuffer> splineDataBuffer{nil};
    id<MTLBuffer> parametersBuffer{nil};
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
    BackendPropagationView& model;
    BackendLikelihoodView& likelihoodModel;
    Result lastResult{};
    BackendTimingSummary buildTiming{};
    BackendTimingSummary lastTiming{};
    std::vector<float> parameterValuesScratch{};
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

    [[nodiscard]] bool buildDeviceModel();
    void updateDeviceParameters(const ParameterSnapshot& parameters_);
    [[nodiscard]] bool encodeEventWeights(id<MTLComputeCommandEncoder> encoder);
    [[nodiscard]] bool encodeCachedDialResponses(id<MTLComputeCommandEncoder> encoder,
                                                 id<MTLComputePipelineState> pipeline_,
                                                 id<MTLBuffer> cachedResponsesBuffer_,
                                                 id<MTLBuffer> descriptorsBuffer_,
                                                 uint32_t descriptorCount_);
    [[nodiscard]] bool encodeHistogramsFromDeviceWeights(id<MTLComputeCommandEncoder> encoder);
    [[nodiscard]] bool runDevicePropagation(const ParameterSnapshot& parameters_, bool needHistograms_);
    void copyDeviceEventWeightsToHostResult();
    [[nodiscard]] bool calculateHistogramsOnDevice();
    void calculateLikelihood();
    void materializeEventWeights();
    void materializeHistograms();
  };

}

namespace {
  constexpr uint32_t kMpsDialTypeNorm{0};
  constexpr uint32_t kMpsDialTypeCompactSpline{1};
  constexpr uint32_t kMpsDialTypeUniformSpline{2};
  constexpr uint32_t kMpsDialTypeMonotonicSpline{3};
  constexpr uint32_t kMpsDialTypeGeneralSpline{4};
  constexpr uint32_t kMpsDialTypeGraph{5};
  constexpr uint32_t kMpsDialFlagAllowExtrapolation{1u << 0};
  constexpr uint32_t kMpsDialFlagCached{1u << 1};
  constexpr uint32_t kMpsCachedDialReuseThreshold{8};

  static NSString* const kMpsBackendMetalSource = GUNDAM_MPS_BACKEND_KERNEL_SOURCE;

  struct MpsEventDialRanges {
    uint32_t normOffset{0};
    uint32_t normCount{0};
    uint32_t compactOffset{0};
    uint32_t compactCount{0};
    uint32_t uniformOffset{0};
    uint32_t uniformCount{0};
    uint32_t monotonicOffset{0};
    uint32_t monotonicCount{0};
    uint32_t generalOffset{0};
    uint32_t generalCount{0};
    uint32_t graphOffset{0};
    uint32_t graphCount{0};
  };

  struct MpsNormDialOccurrence {
    uint32_t parameterIndex{0};
    float minResponse{1.0f};
    float maxResponse{1.0f};
  };

  struct MpsSplineDialDescriptor {
    uint32_t parameterIndex{0};
    uint32_t splineOffset{0};
    uint32_t splineSize{0};
    uint32_t flags{0};
    float minResponse{1.0f};
    float maxResponse{1.0f};
  };

  struct MpsPackedDialRef {
    uint32_t type{0};
    uint32_t localIndex{0};
  };

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

  template<typename T>
  void copyToBuffer(id<MTLBuffer> buffer, const std::vector<T>& values) {
    if( buffer == nil or values.empty() ){ return; }
    std::memcpy(buffer.contents, values.data(), values.size() * sizeof(T));
  }

  [[nodiscard]] inline double secondsSince(std::chrono::steady_clock::time_point start_) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  }

  inline bool isMpsSupportedDialType(const DialBase* dialBase_) {
    return dynamic_cast<const Norm*>(dialBase_) != nullptr
           or dynamic_cast<const CompactSpline*>(dialBase_) != nullptr
           or dynamic_cast<const UniformSpline*>(dialBase_) != nullptr
           or dynamic_cast<const MonotonicSpline*>(dialBase_) != nullptr
           or dynamic_cast<const GeneralSpline*>(dialBase_) != nullptr
           or dynamic_cast<const Graph*>(dialBase_) != nullptr
           or dynamic_cast<const Shift*>(dialBase_) != nullptr;
  }

  inline void appendUnique(std::vector<std::string>& values_, const std::string& value_) {
    if( std::find(values_.begin(), values_.end(), value_) == values_.end() ){
      values_.emplace_back(value_);
    }
  }

  [[nodiscard]] inline std::string joinValues(const std::vector<std::string>& values_) {
    std::string out;
    for( std::size_t iValue = 0 ; iValue < values_.size() ; iValue++ ){
      if( iValue != 0 ){ out += ", "; }
      out += values_[iValue];
    }
    return out;
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
