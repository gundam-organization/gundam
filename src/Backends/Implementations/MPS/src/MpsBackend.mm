#include "MpsBackend.h"

#include "MpsBackendKernelSource.h"

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
#include "Parameter.h"
#include "Shift.h"
#include "UniformSpline.h"

#include "Logger.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {
  constexpr uint32_t kMpsDialTypeNorm{0};
  constexpr uint32_t kMpsDialTypeCompactSpline{1};
  constexpr uint32_t kMpsDialTypeUniformSpline{2};
  constexpr uint32_t kMpsDialTypeMonotonicSpline{3};
  constexpr uint32_t kMpsDialTypeGeneralSpline{4};
  constexpr uint32_t kMpsDialTypeGraph{5};

  static NSString* const kMpsBackendMetalSource = GUNDAM_MPS_BACKEND_KERNEL_SOURCE;

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
  id<MTLBuffer> dialTypesBuffer{nil};
  id<MTLBuffer> dialParameterIndicesBuffer{nil};
  id<MTLBuffer> dialMinResponsesBuffer{nil};
  id<MTLBuffer> dialMaxResponsesBuffer{nil};
  id<MTLBuffer> dialSplineOffsetsBuffer{nil};
  id<MTLBuffer> dialSplineSizesBuffer{nil};
  id<MTLBuffer> dialAllowExtrapolationBuffer{nil};
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

  BackendModel model{};
  BackendLikelihoodModel likelihoodModel{};
  Result lastResult{};
  BackendTimingSummary lastTiming{};
  std::vector<float> parameterValuesScratch{};
  std::string deviceModelFallbackReason{};
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

    eventWeightsPipeline = makePipeline(device, library, @"compute_event_weights");
    histogramPipeline = makePipeline(device, library, @"fill_histograms");
    histogramPartialsPipeline = makePipeline(device, library, @"fill_histogram_partials_by_bin");
    histogramFinalizePipeline = makePipeline(device, library, @"finalize_histograms_from_partials");
    [library release];
    if( eventWeightsPipeline == nil or histogramPipeline == nil
        or histogramPartialsPipeline == nil or histogramFinalizePipeline == nil ){ return; }

    commandQueue = [device newCommandQueue];
    if( commandQueue == nil ){ return; }

    isAvailable = true;
  }

  ~Impl() {
    releaseDeviceBuffers();
    [eventWeightsPipeline release];
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
    releaseBuffer(dialTypesBuffer);
    releaseBuffer(dialParameterIndicesBuffer);
    releaseBuffer(dialMinResponsesBuffer);
    releaseBuffer(dialMaxResponsesBuffer);
    releaseBuffer(dialSplineOffsetsBuffer);
    releaseBuffer(dialSplineSizesBuffer);
    releaseBuffer(dialAllowExtrapolationBuffer);
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

  void resetResult(const PropagationRequest& request_) {
    lastResult.token.id = nextTokenId++;
    lastResult.token.isValid = true;
    lastResult.status = PropagationStatus();
    lastResult.status.backend = BackendStatus::Running;
    lastResult.eventWeights.clear();
    lastResult.histSums.clear();
    lastResult.histSumSquares.clear();
    lastResult.likelihood = 0;
    lastTiming = BackendTimingSummary();

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
    deviceModelFallbackReason.clear();

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

    std::vector<std::string> unsupportedDialTypes;
    for( const auto& eventDial : model.eventDials ){
      const auto* interface = eventDial.interface;
      if( interface == nullptr or interface->getDialBaseRef() == nullptr ){
        return fail("at least one event dial has no DialInterface/DialBase.");
      }
      const auto* dialBase = interface->getDialBaseRef();
      if( not isMpsSupportedDialType(dialBase) ){
        appendUnique(unsupportedDialTypes, dialBase->getDialTypeName());
      }
    }
    if( not unsupportedDialTypes.empty() ){
      return fail("unsupported MPS dial types present in propagator: " + joinValues(unsupportedDialTypes)
                  + ". Supported types are Norm, CompactSpline, UniformSpline, MonotonicSpline, GeneralSpline, Graph and Shift.");
    }

    std::vector<float> baseWeights(model.events.size());
    std::vector<uint32_t> eventDialOffsets(model.events.size());
    std::vector<uint32_t> eventDialCounts(model.events.size());
    std::vector<int> globalBins(model.events.size());
    std::vector<uint32_t> eventsPerBin(model.totalBins, 0);
    std::vector<uint32_t> dialTypes{};
    std::vector<uint32_t> dialParameterIndices{};
    std::vector<float> dialMinResponses{};
    std::vector<float> dialMaxResponses{};
    std::vector<uint32_t> dialSplineOffsets{};
    std::vector<uint32_t> dialSplineSizes{};
    std::vector<uint32_t> dialAllowExtrapolation{};
    std::vector<float> splineData{};
    dialTypes.reserve(model.eventDials.size());
    dialParameterIndices.reserve(model.eventDials.size());
    dialMinResponses.reserve(model.eventDials.size());
    dialMaxResponses.reserve(model.eventDials.size());
    dialSplineOffsets.reserve(model.eventDials.size());
    dialSplineSizes.reserve(model.eventDials.size());
    dialAllowExtrapolation.reserve(model.eventDials.size());

    for( const auto& event : model.events ){
      if( event.globalBinIndex < 0 or event.globalBinIndex >= model.totalBins ){
        return fail("at least one event has an invalid global bin index.");
      }
      baseWeights[event.resultIndex] = float(event.baseWeight);
      eventDialOffsets[event.resultIndex] = uint32_t(dialTypes.size());
      globalBins[event.resultIndex] = event.globalBinIndex;
      eventsPerBin[event.globalBinIndex]++;
      uint32_t packedDialCount = 0;
      for( std::size_t iDial = 0 ; iDial < event.dialCount ; iDial++ ){
        const auto& eventDial = model.eventDials[event.firstDial + iDial];
        const auto* interface = eventDial.interface;
        if( interface == nullptr or interface->getDialBaseRef() == nullptr ){
          return fail("at least one event dial has no DialInterface/DialBase.");
        }
        const auto* dialBase = interface->getDialBaseRef();

        if( const auto* shift = dynamic_cast<const Shift*>(dialBase) ){
          baseWeights[event.resultIndex] *= float(shift->evalResponse(DialInputBuffer()));
          continue;
        }

        const auto* inputBuffer = interface->getInputBufferRef();
        if( inputBuffer == nullptr or inputBuffer->getBufferSize() != 1 ){
          return fail("dial type " + dialBase->getDialTypeName()
                      + " is not MPS-compatible because it does not have exactly one input parameter.");
        }

        int parameterIndex = findParameterIndex(&inputBuffer->getParameter(0));
        if( parameterIndex < 0 ){
          return fail("dial type " + dialBase->getDialTypeName()
                      + " references a parameter missing from the backend parameter table.");
        }

        uint32_t packedDialType = 0;
        uint32_t packedSplineOffset = 0;
        uint32_t packedSplineSize = 0;
        uint32_t packedAllowExtrapolation = 0;

        if( dynamic_cast<const Norm*>(dialBase) != nullptr ){
          packedDialType = kMpsDialTypeNorm;
        }
        else if( dynamic_cast<const CompactSpline*>(dialBase) != nullptr ){
          const auto& data = dialBase->getDialData();
          if( data.size() < 6 ){
            return fail("CompactSpline dial data is too small for MPS evaluation.");
          }
          packedDialType = kMpsDialTypeCompactSpline;
          packedSplineOffset = uint32_t(splineData.size());
          packedSplineSize = uint32_t(data.size());
          packedAllowExtrapolation = dialBase->getAllowExtrapolation() ? 1 : 0;
          splineData.reserve(splineData.size() + data.size());
          for( auto value : data ){ splineData.emplace_back(float(value)); }
        }
        else if( dynamic_cast<const UniformSpline*>(dialBase) != nullptr ){
          const auto& data = dialBase->getDialData();
          if( data.size() < 8 ){
            return fail("UniformSpline dial data is too small for MPS evaluation.");
          }
          packedDialType = kMpsDialTypeUniformSpline;
          packedSplineOffset = uint32_t(splineData.size());
          packedSplineSize = uint32_t(data.size());
          packedAllowExtrapolation = dialBase->getAllowExtrapolation() ? 1 : 0;
          splineData.reserve(splineData.size() + data.size());
          for( auto value : data ){ splineData.emplace_back(float(value)); }
        }
        else if( dynamic_cast<const MonotonicSpline*>(dialBase) != nullptr ){
          const auto& data = dialBase->getDialData();
          if( data.size() < 5 ){
            return fail("MonotonicSpline dial data is too small for MPS evaluation.");
          }
          packedDialType = kMpsDialTypeMonotonicSpline;
          packedSplineOffset = uint32_t(splineData.size());
          packedSplineSize = uint32_t(data.size());
          packedAllowExtrapolation = dialBase->getAllowExtrapolation() ? 1 : 0;
          splineData.reserve(splineData.size() + data.size());
          for( auto value : data ){ splineData.emplace_back(float(value)); }
        }
        else if( dynamic_cast<const GeneralSpline*>(dialBase) != nullptr ){
          const auto& data = dialBase->getDialData();
          if( data.size() < 11 ){
            return fail("GeneralSpline dial data is too small for MPS evaluation.");
          }
          packedDialType = kMpsDialTypeGeneralSpline;
          packedSplineOffset = uint32_t(splineData.size());
          packedSplineSize = uint32_t(data.size());
          packedAllowExtrapolation = dialBase->getAllowExtrapolation() ? 1 : 0;
          splineData.reserve(splineData.size() + data.size());
          for( auto value : data ){ splineData.emplace_back(float(value)); }
        }
        else if( dynamic_cast<const Graph*>(dialBase) != nullptr ){
          const auto& data = dialBase->getDialData();
          if( data.size() < 2 ){
            return fail("Graph dial data is too small for MPS evaluation.");
          }
          packedDialType = kMpsDialTypeGraph;
          packedSplineOffset = uint32_t(splineData.size());
          packedSplineSize = uint32_t(data.size());
          packedAllowExtrapolation = dialBase->getAllowExtrapolation() ? 1 : 0;
          splineData.reserve(splineData.size() + data.size());
          for( auto value : data ){ splineData.emplace_back(float(value)); }
        }
        else{
          return fail("dial type " + dialBase->getDialTypeName()
                      + " unexpectedly passed the MPS compatibility scan but has no device encoder.");
        }

        float minResponse = -std::numeric_limits<float>::infinity();
        float maxResponse = std::numeric_limits<float>::infinity();
        const auto* supervisor = interface->getResponseSupervisorRef();
        if( supervisor != nullptr ){
          if( not std::isnan(supervisor->getMinResponse()) ){
            minResponse = float(supervisor->getMinResponse());
          }
          if( not std::isnan(supervisor->getMaxResponse()) ){
            maxResponse = float(supervisor->getMaxResponse());
          }
        }

        dialTypes.emplace_back(packedDialType);
        dialParameterIndices.emplace_back(uint32_t(parameterIndex));
        dialMinResponses.emplace_back(minResponse);
        dialMaxResponses.emplace_back(maxResponse);
        dialSplineOffsets.emplace_back(packedSplineOffset);
        dialSplineSizes.emplace_back(packedSplineSize);
        dialAllowExtrapolation.emplace_back(packedAllowExtrapolation);
        packedDialCount++;
      }
      eventDialCounts[event.resultIndex] = packedDialCount;
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

    if( dialTypes.empty() ){
      dialTypes.emplace_back(0);
      dialParameterIndices.emplace_back(0);
      dialMinResponses.emplace_back(1.0f);
      dialMaxResponses.emplace_back(1.0f);
      dialSplineOffsets.emplace_back(0);
      dialSplineSizes.emplace_back(0);
      dialAllowExtrapolation.emplace_back(0);
    }
    if( splineData.empty() ){ splineData.emplace_back(0); }

    uint32_t nEvents = uint32_t(model.events.size());
    uint32_t totalBins = uint32_t(model.totalBins);
    uint32_t chunkSize = histogramChunkSize;
    uint32_t totalPartials = totalBins * maxHistogramChunksPerBin;

    baseWeightsBuffer = makePrivateBuffer(device, commandQueue, baseWeights);
    eventDialOffsetsBuffer = makePrivateBuffer(device, commandQueue, eventDialOffsets);
    eventDialCountsBuffer = makePrivateBuffer(device, commandQueue, eventDialCounts);
    globalBinsBuffer = makePrivateBuffer(device, commandQueue, globalBins);
    binEventOffsetsBuffer = makePrivateBuffer(device, commandQueue, binEventOffsets);
    binEventIndicesBuffer = makePrivateBuffer(device, commandQueue, binEventIndices);
    dialTypesBuffer = makePrivateBuffer(device, commandQueue, dialTypes);
    dialParameterIndicesBuffer = makePrivateBuffer(device, commandQueue, dialParameterIndices);
    dialMinResponsesBuffer = makePrivateBuffer(device, commandQueue, dialMinResponses);
    dialMaxResponsesBuffer = makePrivateBuffer(device, commandQueue, dialMaxResponses);
    dialSplineOffsetsBuffer = makePrivateBuffer(device, commandQueue, dialSplineOffsets);
    dialSplineSizesBuffer = makePrivateBuffer(device, commandQueue, dialSplineSizes);
    dialAllowExtrapolationBuffer = makePrivateBuffer(device, commandQueue, dialAllowExtrapolation);
    splineDataBuffer = makePrivateBuffer(device, commandQueue, splineData);
    eventWeightsBuffer = makePrivateEmptyBuffer(device, model.events.size() * sizeof(float));
    eventWeightsReadbackBuffer = makeSharedEmptyBuffer(device, model.events.size() * sizeof(float));
    parametersBuffer = makeSharedEmptyBuffer(device, std::max<std::size_t>(1, model.parameters.size()) * sizeof(float));
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

    if( baseWeightsBuffer == nil or eventDialOffsetsBuffer == nil or eventDialCountsBuffer == nil
        or globalBinsBuffer == nil or binEventOffsetsBuffer == nil or binEventIndicesBuffer == nil
        or dialTypesBuffer == nil or dialParameterIndicesBuffer == nil
        or dialMinResponsesBuffer == nil or dialMaxResponsesBuffer == nil
        or dialSplineOffsetsBuffer == nil or dialSplineSizesBuffer == nil
        or dialAllowExtrapolationBuffer == nil or splineDataBuffer == nil
        or eventWeightsBuffer == nil or parametersBuffer == nil
        or partialHistSumsBuffer == nil or partialHistSumSquaresBuffer == nil
        or histSumsBuffer == nil or histSumSquaresBuffer == nil or eventWeightsReadbackBuffer == nil
        or histSumsReadbackBuffer == nil or histSumSquaresReadbackBuffer == nil or nEventsBuffer == nil
        or totalBinsBuffer == nil or maxHistogramChunksPerBinBuffer == nil
        or histogramChunkSizeBuffer == nil ){
      releaseDeviceBuffers();
      return fail("Metal buffer allocation failed while building the MPS backend model.");
    }

    parameterValuesScratch.resize(model.parameters.size());
    isDeviceModelSupported = true;
    return true;
  }

  void updateDeviceParameters() {
    auto start = std::chrono::steady_clock::now();
    if( parameterValuesScratch.size() != model.parameters.size() ){
      parameterValuesScratch.resize(model.parameters.size());
    }
    for( std::size_t iPar = 0 ; iPar < model.parameters.size() ; iPar++ ){
      parameterValuesScratch[iPar] = float(model.parameters[iPar]->getParameterValue());
    }
    copyToBuffer(parametersBuffer, parameterValuesScratch);
    lastTiming.parameterUploadSeconds += secondsSince(start);
  }

  bool encodeEventWeights(id<MTLComputeCommandEncoder> encoder) {
    if( not isDeviceModelSupported ){ return false; }
    [encoder setComputePipelineState:eventWeightsPipeline];
    [encoder setBuffer:eventWeightsBuffer offset:0 atIndex:0];
    [encoder setBuffer:baseWeightsBuffer offset:0 atIndex:1];
    [encoder setBuffer:eventDialOffsetsBuffer offset:0 atIndex:2];
    [encoder setBuffer:eventDialCountsBuffer offset:0 atIndex:3];
    [encoder setBuffer:dialTypesBuffer offset:0 atIndex:4];
    [encoder setBuffer:dialParameterIndicesBuffer offset:0 atIndex:5];
    [encoder setBuffer:dialMinResponsesBuffer offset:0 atIndex:6];
    [encoder setBuffer:dialMaxResponsesBuffer offset:0 atIndex:7];
    [encoder setBuffer:dialSplineOffsetsBuffer offset:0 atIndex:8];
    [encoder setBuffer:dialSplineSizesBuffer offset:0 atIndex:9];
    [encoder setBuffer:dialAllowExtrapolationBuffer offset:0 atIndex:10];
    [encoder setBuffer:splineDataBuffer offset:0 atIndex:11];
    [encoder setBuffer:parametersBuffer offset:0 atIndex:12];
    [encoder setBuffer:nEventsBuffer offset:0 atIndex:13];

    NSUInteger width = std::min<NSUInteger>(eventWeightsPipeline.maxTotalThreadsPerThreadgroup, 256);
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

    auto encodeStart = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
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
    if( lastResult.eventWeights.empty() and eventWeightsBuffer != nil ){
      copyDeviceEventWeightsToHostResult();
    }
    auto start = std::chrono::steady_clock::now();
    LogThrowIf(lastResult.eventWeights.size() != model.events.size());
    for( const auto& event : model.events ){
      event.event->getWeights().current = lastResult.eventWeights[event.resultIndex];
    }
    lastTiming.eventWeightMaterializationSeconds += secondsSince(start);
  }

  void materializeHistograms() {
    auto start = std::chrono::steady_clock::now();
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

void Backends::MpsBackend::build(const BackendModel& model_) {
  _impl_->model = model_;
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
