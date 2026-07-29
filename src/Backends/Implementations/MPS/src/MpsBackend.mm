#include "MpsBackend.h"

#include "DialInputBuffer.h"
#include "DialInterface.h"
#include "Event.h"
#include "Histogram.h"
#include "Parameter.h"

#include "Logger.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {
  static NSString* const kMpsBackendMetalSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

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
)METAL";

  template<typename T>
  id<MTLBuffer> makeBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    if( values.empty() ){ return nil; }
    return [device newBufferWithBytes:values.data()
                               length:values.size() * sizeof(T)
                              options:MTLResourceStorageModeShared];
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
  id<MTLComputePipelineState> histogramPipeline{nil};
  bool isAvailable{false};

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

    id<MTLFunction> histogramFunction = [library newFunctionWithName:@"fill_histograms"];
    if( histogramFunction == nil ){
      LogError << "Could not find fill_histograms Metal function." << std::endl;
      [library release];
      return;
    }

    histogramPipeline = [device newComputePipelineStateWithFunction:histogramFunction error:&error];
    [histogramFunction release];
    [library release];
    if( histogramPipeline == nil ){
      if( error != nil ){
        LogError << "Could not create MPS backend histogram pipeline: "
                 << [[error localizedDescription] UTF8String] << std::endl;
      }
      return;
    }

    commandQueue = [device newCommandQueue];
    if( commandQueue == nil ){ return; }

    isAvailable = true;
  }

  ~Impl() {
    [histogramPipeline release];
    [commandQueue release];
    [device release];
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

  if( request_.has(OutputRequest::EventWeights) ){
    _impl_->calculateEventWeights();
    _impl_->lastResult.status.eventWeights = OutputState::ReadyOnHost;
  }

  if( request_.has(OutputRequest::Histograms) or request_.has(OutputRequest::Likelihood) ){
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
