#include "MpsBackend.h"

#include "MpsBackendKernelSource.h"

#include "CalculateCompactSpline.h"
#include "CalculateGeneralSpline.h"
#include "CalculateGraph.h"
#include "CalculateMonotonicSpline.h"
#include "CalculateUniformSpline.h"

#include "DialInputBuffer.h"
#include "DialInterface.h"
#include "DialResponseSupervisor.h"
#include "Event.h"
#include "GeneralSpline.h"
#include "Graph.h"
#include "Histogram.h"
#include "CompactSpline.h"
#include "MonotonicSpline.h"
#include "Norm.h"
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

  id<MTLBuffer> makeSharedEmptyBuffer(id<MTLDevice> device, std::size_t byteSize) {
    if( byteSize == 0 ){ return nil; }
    return [device newBufferWithLength:byteSize options:MTLResourceStorageModeShared];
  }

  id<MTLBuffer> makePrivateEmptyBuffer(id<MTLDevice> device, std::size_t byteSize) {
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

  void releaseBuffer(id<MTLBuffer>& buffer) {
    [buffer release];
    buffer = nil;
  }

  template<typename T>
  void copyToBuffer(id<MTLBuffer> buffer, const std::vector<T>& values) {
    if( buffer == nil or values.empty() ){ return; }
    std::memcpy(buffer.contents, values.data(), values.size() * sizeof(T));
  }

  [[nodiscard]] double secondsSince(std::chrono::steady_clock::time_point start_) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  }

  bool isMpsSupportedDialType(const DialBase* dialBase_) {
    return dynamic_cast<const Norm*>(dialBase_) != nullptr
           or dynamic_cast<const CompactSpline*>(dialBase_) != nullptr
           or dynamic_cast<const UniformSpline*>(dialBase_) != nullptr
           or dynamic_cast<const MonotonicSpline*>(dialBase_) != nullptr
           or dynamic_cast<const GeneralSpline*>(dialBase_) != nullptr
           or dynamic_cast<const Graph*>(dialBase_) != nullptr
           or dynamic_cast<const Shift*>(dialBase_) != nullptr;
  }

  void appendUnique(std::vector<std::string>& values_, const std::string& value_) {
    if( std::find(values_.begin(), values_.end(), value_) == values_.end() ){
      values_.emplace_back(value_);
    }
  }

  std::string joinValues(const std::vector<std::string>& values_) {
    std::string out;
    for( std::size_t iValue = 0 ; iValue < values_.size() ; iValue++ ){
      if( iValue != 0 ){ out += ", "; }
      out += values_[iValue];
    }
    return out;
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

  BackendEngineView engineView{};
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
  mutable DialInputBuffer scratchDialInputBuffer{};

  Impl() : model(engineView.propagation), likelihoodModel(engineView.likelihood) {
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

  ~Impl() {
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

  void releaseDeviceBuffers() {
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

  bool isCurrentToken(const PropagationToken& token_) const {
    return token_.isValid and lastResult.token.isValid and token_.id == lastResult.token.id;
  }

  void resetResult() {
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

  double getDialInputValue(const BackendDialInputRef& inputRef_, const ParameterSnapshot& parameters_) const {
    LogThrowIf(parameters_.empty(), "MpsBackend requires a populated ParameterSnapshot.");
    return parameters_.values.at(inputRef_.parameterIndex);
  }

  static double applyDialInputTransform(const BackendDialInputRef& inputRef_, double rawValue_) {
    if( not inputRef_.useMirror ){ return rawValue_; }

    double transformed = std::abs(std::fmod(
        rawValue_ - inputRef_.mirrorMin,
        2 * inputRef_.mirrorRange
    ));

    if( transformed > inputRef_.mirrorRange ){
      transformed -= 2 * inputRef_.mirrorRange;
      transformed = -transformed;
    }

    return transformed + inputRef_.mirrorMin;
  }

  double evaluateDialResponse(const BackendDialRef& dialRef_, const ParameterSnapshot& parameters_) const {
    auto clampResponse = [&dialRef_](double response_){
      if( dialRef_.hasMinResponse and response_ < dialRef_.minResponse ){ response_ = dialRef_.minResponse; }
      if( dialRef_.hasMaxResponse and response_ > dialRef_.maxResponse ){ response_ = dialRef_.maxResponse; }
      return response_;
    };

    auto getInput = [this, &dialRef_, &parameters_](std::size_t iInput_){
      const auto& inputRef = model.dialInputs.at(dialRef_.firstInput + iInput_);
      return applyDialInputTransform(inputRef, getDialInputValue(inputRef, parameters_));
    };

    const double* payload = model.dialPayloads.data() + dialRef_.payloadOffset;

    switch( dialRef_.type ){
      case BackendDialType::Norm:
        return clampResponse(getInput(0));
      case BackendDialType::Shift:
        return clampResponse(payload[0]);
      case BackendDialType::CompactSpline: {
        double x = getInput(0);
        if( not dialRef_.allowExtrapolation ){
          x = std::clamp(x, payload[0], payload[0] + payload[1] * double(dialRef_.payloadSize - 3));
        }
        return clampResponse(CalculateCompactSpline(x, -1E20, 1E20, payload, int(dialRef_.payloadSize - 2)));
      }
      case BackendDialType::UniformSpline: {
        double x = getInput(0);
        if( not dialRef_.allowExtrapolation ){
          x = std::clamp(x, payload[0], payload[0] + payload[1] * double((dialRef_.payloadSize - 2) / 2 - 1));
        }
        return clampResponse(CalculateUniformSpline(x, -1E20, 1E20, payload, int(dialRef_.payloadSize)));
      }
      case BackendDialType::MonotonicSpline: {
        double x = getInput(0);
        if( not dialRef_.allowExtrapolation ){
          x = std::clamp(x, payload[0], payload[0] + payload[1] * double(dialRef_.payloadSize - 3));
        }
        return clampResponse(CalculateMonotonicSpline(x, -1E20, 1E20, payload, int(dialRef_.payloadSize - 2)));
      }
      case BackendDialType::GeneralSpline: {
        double x = getInput(0);
        if( not dialRef_.allowExtrapolation ){
          x = std::clamp(x, payload[0], payload[0] + payload[1] * double((dialRef_.payloadSize - 2) / 3 - 1));
        }
        return clampResponse(CalculateGeneralSpline(x, -1E20, 1E20, payload, int(dialRef_.payloadSize)));
      }
      case BackendDialType::Graph: {
        double x = getInput(0);
        if( not dialRef_.allowExtrapolation ){
          x = std::clamp(x, payload[1], payload[dialRef_.payloadSize - 1]);
        }
        return clampResponse(CalculateGraph(x, -1E20, 1E20, payload, int(dialRef_.payloadSize)));
      }
    }

    LogThrow("Unhandled backend dial type in MPS fallback.");
  }

  bool buildDeviceModel() {
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
    auto fillMinMax = [](const BackendDialRef& dialRef_, float& minResponse_, float& maxResponse_) {
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
        auto eventDialIt = std::find_if(model.eventDials.begin(), model.eventDials.end(), [&](const auto& ref_){
          return ref_.payloadOffset == dialOffsets_[iUniqueDial];
        });
        LogThrowIf(eventDialIt == model.eventDials.end(), "Internal MPS packing error: could not resolve unique backend dial descriptor.");
        const auto& eventDial = *eventDialIt;

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

  void updateDeviceParameters(const ParameterSnapshot& parameters_) {
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

  bool encodeEventWeights(id<MTLComputeCommandEncoder> encoder) {
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

  bool encodeCachedDialResponses(id<MTLComputeCommandEncoder> encoder,
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

  bool runDevicePropagation(const ParameterSnapshot& parameters_, bool needHistograms_) {
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
        [blitEncoder copyFromBuffer:histSumsBuffer
                       sourceOffset:0
                           toBuffer:histSumsReadbackBuffer
                  destinationOffset:0
                               size:histogramBytes];
        [blitEncoder copyFromBuffer:histSumSquaresBuffer
                       sourceOffset:0
                           toBuffer:histSumSquaresReadbackBuffer
                  destinationOffset:0
                               size:histogramBytes];
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
      [blitEncoder copyFromBuffer:histSumsBuffer
                     sourceOffset:0
                         toBuffer:histSumsReadbackBuffer
                destinationOffset:0
                             size:histogramBytes];
      [blitEncoder copyFromBuffer:histSumSquaresBuffer
                     sourceOffset:0
                         toBuffer:histSumSquaresReadbackBuffer
                destinationOffset:0
                             size:histogramBytes];
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

  void copyDeviceEventWeightsToHostResult() {
    LogThrowIf(eventWeightsBuffer == nil);
    auto start = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
    std::size_t byteSize = model.events.size() * sizeof(float);
    [encoder copyFromBuffer:eventWeightsBuffer
               sourceOffset:0
                   toBuffer:eventWeightsReadbackBuffer
          destinationOffset:0
                       size:byteSize];
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

  void calculateEventWeights(const ParameterSnapshot& parameters_) {
    lastResult.eventWeights.resize(model.events.size());

    for( const auto& event : model.events ){
      double weight = event.baseWeight;

      for( std::size_t iDial = 0 ; iDial < event.dialCount ; iDial++ ){
        const auto& dialRef = model.eventDials[event.firstDial + iDial];
        weight *= evaluateDialResponse(dialRef, parameters_);
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
      calculateEventWeights(ParameterSnapshot{});
    }

    std::vector<float> eventWeightsFloat(lastResult.eventWeights.size());
    std::vector<int> globalBins(model.events.size());
    for( const auto& event : model.events ){
      eventWeightsFloat[event.resultIndex] = float(lastResult.eventWeights[event.resultIndex]);
      globalBins[event.resultIndex] = event.globalBinIndex;
    }

    auto eventWeightsBuffer = makeSharedBuffer(device, eventWeightsFloat);
    auto globalBinsBuffer = makeSharedBuffer(device, globalBins);
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
    auto start = std::chrono::steady_clock::now();
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
    lastTiming.likelihoodHostSeconds += secondsSince(start);
  }

  void materializeEventWeights() {
    auto start = std::chrono::steady_clock::now();
    if( lastResult.eventWeights.empty() and eventWeightsBuffer != nil ){
      copyDeviceEventWeightsToHostResult();
    }
    LogThrowIf(lastResult.eventWeights.size() != model.events.size());
    lastTiming.eventWeightMaterializationSeconds += secondsSince(start);
  }

  void materializeHistograms() {
    auto start = std::chrono::steady_clock::now();
    LogThrowIf(lastResult.histSums.size() != std::size_t(model.totalBins));
    LogThrowIf(lastResult.histSumSquares.size() != std::size_t(model.totalBins));
    lastTiming.histogramMaterializationSeconds += secondsSince(start);
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

void Backends::MpsBackend::build(const BackendEngineView& engineView_) {
  _impl_->engineView = engineView_;
  _impl_->lastResult = Impl::Result();
  if( not _impl_->buildDeviceModel() ){
    LogWarning << "MPS backend cannot use the GPU propagation path: "
               << (_impl_->deviceModelFallbackReason.empty() ? "unknown compatibility issue."
                                                             : _impl_->deviceModelFallbackReason)
               << " Falling back to the standard backend path for unsupported calculations."
               << std::endl;
  }
  _impl_->isBuilt = true;
}

Backends::PropagationToken Backends::MpsBackend::requestPropagation(const ParameterSnapshot& parameters_) {
  LogThrowIf(not _impl_->isBuilt, "MpsBackend has not been built.");
  LogThrowIf(not parameters_.empty() and parameters_.values.size() != _impl_->model.parameterCount,
             "ParameterSnapshot size mismatch: " << parameters_.values.size()
                                                 << " != " << _impl_->model.parameterCount);

  _impl_->resetResult();

  if( not _impl_->isAvailable ){
    _impl_->lastResult.status.backend = BackendStatus::Unavailable;
    _impl_->lastResult.status.eventWeights = OutputState::Failed;
    _impl_->lastResult.status.histograms = OutputState::Failed;
    _impl_->lastResult.status.sampleLikelihoods = OutputState::Failed;
    _impl_->lastResult.status.statLikelihood = OutputState::Failed;
    _impl_->lastResult.token.isValid = false;
    return {};
  }

  bool needsEventWeights = true;
  bool needsHistograms = true;
  bool usedDevicePropagation = false;

  if( needsEventWeights or needsHistograms ){
    usedDevicePropagation = _impl_->runDevicePropagation(parameters_, needsHistograms);
    if( usedDevicePropagation ){
      _impl_->lastResult.status.eventWeights = OutputState::ReadyOnDevice;
      _impl_->lastResult.status.histograms = OutputState::ReadyOnDevice;
    }
  }

  if( not usedDevicePropagation ){
    _impl_->calculateEventWeights(parameters_);
    _impl_->lastResult.status.eventWeights = OutputState::ReadyOnHost;
  }

  if( needsHistograms and not usedDevicePropagation ){
    if( not _impl_->calculateHistogramsOnDevice() ){
      _impl_->lastResult.status.backend = BackendStatus::Failed;
      _impl_->lastResult.status.histograms = OutputState::Failed;
      _impl_->lastResult.status.statLikelihood = OutputState::Failed;
      return _impl_->lastResult.token;
    }
    _impl_->lastResult.status.histograms = OutputState::ReadyOnDevice;
  }

  _impl_->lastResult.status.sampleLikelihoods = OutputState::Failed;
  if( _impl_->likelihoodModel.empty() ){
    _impl_->lastResult.status.statLikelihood = OutputState::Failed;
  }
  else{
    _impl_->calculateLikelihood();
    _impl_->lastResult.status.statLikelihood = OutputState::ReadyOnHost;
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

const Backends::BackendEngineView& Backends::MpsBackend::getEngineView() const {
  return _impl_->engineView;
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
  else if( output_ == OutputRequest::SampleLikelihoods ){
    LogThrow("MpsBackend cannot materialize sample likelihoods yet.");
  }
  else if( output_ == OutputRequest::StatLikelihood ){
    _impl_->lastResult.status.statLikelihood = OutputState::ReadyOnHost;
  }
  else{
    LogThrow("MpsBackend cannot materialize requested output yet.");
  }
}

double Backends::MpsBackend::getLikelihood(const PropagationToken& token_) const {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
  LogThrowIf(_impl_->lastResult.status.statLikelihood != OutputState::ReadyOnDevice
             and _impl_->lastResult.status.statLikelihood != OutputState::ReadyOnHost,
             "Backend likelihood is not ready.");
  return _impl_->lastResult.likelihood;
}

const std::vector<double>& Backends::MpsBackend::getEventWeightsHostView(const PropagationToken& token_) const {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
  return _impl_->lastResult.eventWeights;
}

const std::vector<double>& Backends::MpsBackend::getHistogramSumsHostView(const PropagationToken& token_) const {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
  return _impl_->lastResult.histSums;
}

const std::vector<double>& Backends::MpsBackend::getHistogramSumSquaresHostView(const PropagationToken& token_) const {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
  return _impl_->lastResult.histSumSquares;
}

Backends::BackendDeviceView Backends::MpsBackend::getDeviceView(const PropagationToken& token_) const {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
  BackendDeviceView out;
  out.device = _impl_->device;
  out.eventWeights = _impl_->eventWeightsBuffer;
  out.eventWeightsBytes = _impl_->model.events.size() * sizeof(float);
  out.histSums = _impl_->histSumsBuffer;
  out.histSumSquares = _impl_->histSumSquaresBuffer;
  out.histogramBytes = std::size_t(_impl_->model.totalBins) * sizeof(float);
  return out;
}

Backends::BackendTimingSummary Backends::MpsBackend::getLastTimingSummary() const {
  return _impl_->lastTiming;
}
