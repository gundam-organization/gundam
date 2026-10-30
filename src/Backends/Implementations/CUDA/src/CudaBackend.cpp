#include "CudaBackend.h"
#include "CudaWeightModel.h"
#include "PropagationResult.h"
#include "LikelihoodModel.h"
#include "Semantics/BackendHostPropagation.h"

#ifdef GUNDAM_USING_CUDA
#include "CudaDeviceContext.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
  using Clock = std::chrono::steady_clock;
  double elapsed(Clock::time_point start_) { return std::chrono::duration<double>(Clock::now() - start_).count(); }
}

struct Backends::CudaBackendImpl {
  std::shared_ptr<const EngineView> engineView{};
  PropagationResult result{};
  EventState eventState{};
  bool hasEventState{false};
  bool hasSmoothedBinning{false};
  BackendTimingSummary timing{};
  BackendTimingSummary buildTiming{};
  BackendTimingTotals totals{};
  JsonType diagnostics{{"available", false}};
#ifdef GUNDAM_USING_CUDA
  std::unique_ptr<CudaDeviceContext> context{};
#endif
};

Backends::CudaBackend::CudaBackend() : _impl_(std::make_unique<CudaBackendImpl>()) {}
Backends::CudaBackend::~CudaBackend() = default;

bool Backends::CudaBackend::isCompiled() {
#ifdef GUNDAM_USING_CUDA
  return true;
#else
  return false;
#endif
}

std::string Backends::CudaBackend::getName() const { return "CUDA/" + _propagationMode_.toString(); }

void Backends::CudaBackend::configureImpl() {
  _impl_ = std::make_unique<CudaBackendImpl>();
  _deviceIndex_ = 0;
  _likelihoodBlockSize_ = 0;
  _reusePrediction_ = true;
  _propagationMode_ = CudaPropagationMode::cachedResponses;
  _weightLayout_ = CudaWeightLayout::warpInterleaved;
  _config_.clearFields();
  _config_.defineFields({{"name"}, {"deviceIndex"}, {"propagationMode"}, {"weightLayout"}, {"likelihoodBlockSize"}, {"reusePrediction"}});
  _config_.checkConfiguration();
  // fillValue tolerates conversion errors; explicit device/mode selections must
  // not silently fall back to defaults when they are malformed.
  if( const auto* deviceIndex = _config_.getJsonEntry("deviceIndex") ){
    if( not deviceIndex->is_number_integer() or *deviceIndex < 0
        or *deviceIndex > std::numeric_limits<int>::max() ){
      throw std::runtime_error("CudaBackend: deviceIndex must be a nonnegative integer within the int range.");
    }
  }
  if( const auto* mode = _config_.getJsonEntry("propagationMode") ){
    if( not mode->is_string() ){ throw std::runtime_error("CudaBackend: propagationMode must be a string."); }
  }
  if( const auto* size = _config_.getJsonEntry("likelihoodBlockSize") ){
    if( not size->is_number_integer() or (*size != 0 and *size != 32 and *size != 64 and *size != 128 and *size != 256) ){
      throw std::runtime_error("CudaBackend: likelihoodBlockSize must be 0 (automatic), 32, 64, 128 or 256.");
    }
  }
  if( const auto* reuse = _config_.getJsonEntry("reusePrediction") ){
    if( not reuse->is_boolean() ){ throw std::runtime_error("CudaBackend: reusePrediction must be a boolean."); }
  }
  _config_.fillValue(_likelihoodBlockSize_, "likelihoodBlockSize");
  _config_.fillValue(_reusePrediction_, "reusePrediction");
  _config_.fillValue(_deviceIndex_, "deviceIndex");
  _config_.fillEnum(_propagationMode_, "propagationMode");
  if( const auto* layout = _config_.getJsonEntry("weightLayout") ){
    if( not layout->is_string() ){ throw std::runtime_error("CudaBackend: weightLayout must be a string."); }
  }
  _config_.fillEnum(_weightLayout_, "weightLayout");
  if( _weightLayout_ != CudaWeightLayout::eventMajor and _weightLayout_ != CudaWeightLayout::warpInterleaved ){
    throw std::runtime_error("CudaBackend: weightLayout must be eventMajor or warpInterleaved.");
  }
  if( _propagationMode_ != CudaPropagationMode::cachedResponses and _propagationMode_ != CudaPropagationMode::recomputeResponses ){
    throw std::runtime_error("CudaBackend: propagationMode must be cachedResponses or recomputeResponses.");
  }
}

void Backends::CudaBackend::initializeDevice() {
#ifdef GUNDAM_USING_CUDA
  if( not _impl_->context ){ _impl_->context = std::make_unique<CudaDeviceContext>(_deviceIndex_); }
#else
  throw std::runtime_error("CUDA support was not compiled. Configure GUNDAM with -DWITH_CUDA_LIB=ON and a CUDA toolkit.");
#endif
}

void Backends::CudaBackend::initializeImpl() { initializeDevice(); }

Backends::BackendCapabilities Backends::CudaBackend::getCapabilities() const {
  BackendCapabilities out;
  out.supportsGpu = isCompiled();
  out.supportsEventWeights = isCompiled();
  out.supportsHistograms = isCompiled();
  out.supportsLikelihood = isCompiled();
  out.supportsObservableTransforms = isCompiled();
  out.supportsDynamicBinning = isCompiled();
  out.deviceName = isCompiled() ? "CUDA not initialized" : "CUDA support not compiled";
#ifdef GUNDAM_USING_CUDA
  if( _impl_->context ){ out.deviceName = _impl_->context->getDeviceName(); }
#endif
  return out;
}

const Backends::EngineView& Backends::CudaBackend::getEngineView() const {
  if( not _impl_->engineView ){ throw std::runtime_error("CudaBackend has not been built."); }
  return *_impl_->engineView;
}

void Backends::CudaBackend::build(const EngineView& engineView_) { buildShared(std::make_shared<EngineView>(engineView_)); }

void Backends::CudaBackend::buildShared(std::shared_ptr<const EngineView> engineView_) {
  invalidateResult();
  _impl_->engineView.reset();
  _impl_->hasEventState = false;
  _impl_->hasSmoothedBinning = false;
  _impl_->eventState = {};
  _impl_->buildTiming = {};
  _impl_->timing = {};
  _impl_->totals = {};
  _impl_->diagnostics = {{"available", false}};
  if( not engineView_ ){ throw std::runtime_error("CudaBackend requires a model."); }
  const auto start = Clock::now();
  const auto packed = makeCudaWeightModel(*engineView_, _weightLayout_ == CudaWeightLayout::warpInterleaved);
  _impl_->buildTiming.buildCompatibilityScanSeconds = elapsed(start);
  // Smaller blocks expose more independent work for short binned likelihoods.
  // This is a starting policy, not a V100 performance claim; benchmark overrides
  // allow measuring occupancy/register tradeoffs without changing arithmetic.
  const unsigned int likelihoodBlockSize = _likelihoodBlockSize_ != 0 ? _likelihoodBlockSize_
      : (engineView_->propagation.totalBins <= 8192 ? 64u : 256u);
  initializeDevice();
#ifdef GUNDAM_USING_CUDA
  const auto uploadStart = Clock::now();
  _impl_->context->build(packed, engineView_->propagation, engineView_->likelihood, likelihoodBlockSize);
  _impl_->buildTiming.buildBufferUploadSeconds = elapsed(uploadStart);
#endif
  _impl_->engineView = std::move(engineView_);
  _impl_->hasEventState = packed.hasEventState;
  _impl_->hasSmoothedBinning = packed.hasSmoothedBinning;
  _impl_->result.outsideSums.assign(packed.samples.size(), 0.);
  _impl_->result.outsideSumSquares.assign(packed.samples.size(), 0.);
  _impl_->diagnostics = {
      {"likelihoodExecution", hasPortableLikelihood(getEngineView().likelihood) ? "device" : "host"},
      {"requestedLikelihoodBlockSize", _likelihoodBlockSize_}, {"likelihoodBlockSize", likelihoodBlockSize},
      {"likelihoodReduction", "parallel256"}, {"reusePrediction", _reusePrediction_},
      {"available", true}, {"events", packed.events.size()}, {"uniqueDials", packed.dials.size()},
      {"graphDials", std::count_if(packed.dials.begin(), packed.dials.end(),
          [](const auto& dial_){ return dial_.type == BackendDialType::Graph; })},
      {"surfaceDials", packed.dials.size() - packed.surfaceOffset},
      {"formulaPrograms", packed.formulaPrograms.size()},
      {"formulaInstructions", packed.formulaInstructions.size()},
      {"compositePrograms", packed.compositePrograms.size()},
      {"compositeInstructions", packed.compositeInstructions.size()},
      {"compositeInstances", packed.compositeInstances.size()},
      {"compositeNodeDataBytes", packed.compositeNodeData.size() * sizeof(std::size_t)},
      {"originalCompositeInstructions", getEngineView().propagation.compositeInstructions.size()},
      {"compositeTemplateBytes", packed.compositePrograms.size() * sizeof(NativeCompositeTemplate)
          + packed.compositeInstructions.size() * sizeof(CompositeInstruction)
          + packed.compositeInputs.size() * sizeof(BackendDialInputDescriptor)},
      {"fastCompositeInstances", std::count_if(packed.compositeInstances.begin(), packed.compositeInstances.end(),
          [&](const auto& instance_){ return packed.compositePrograms[instance_.programIndex].isChain; })},
      {"fastFormulaPrograms", std::count_if(packed.formulaPrograms.begin(), packed.formulaPrograms.end(),
          [](const auto& program_){ return program_.nativePlan.kernel != NativeFormulaKernel::Interpreter; })},
      {"programScratchBytes", packed.programScratchSize * sizeof(double)},
      {"programKernel", packed.onlyFastPrograms ? "fast" : (packed.compositePrograms.empty() and packed.formulaPrograms.empty() ? "none" : "mixed")},
      {"multiInputDependencyGroups", packed.multiInputGroups.size()},
      {"observableResponses", packed.variableResponses.size()},
      {"hasEventState", packed.hasEventState},
      {"hasSmoothedBinning", packed.hasSmoothedBinning},
      {"surfaceDependencyGroups", packed.surfaceGroups.size()},
      {"externalDials", packed.externalDialCount},
      {"externalWeightBlocks", packed.externalWeightBlocks.size()},
      {"tableTerms", packed.tableTerms.size()},
      {"requestedWeightLayout", _weightLayout_.toString()},
      {"weightLayout", packed.eventDialStride == 32 ? "warpInterleaved" : "eventMajor"},
      {"originalEventDialIndices", getEngineView().propagation.eventDialIndices.size()},
      {"packedEventDialIndices", packed.eventDialIndices.size()},
      {"packedEventDialIndexBytes", packed.eventDialIndices.size() * sizeof(std::uint32_t)},
      {"device", getCapabilities().deviceName}};
  _impl_->timing = _impl_->buildTiming;
}

std::string Backends::CudaBackend::getDiagnostics() const { return _impl_->diagnostics.dump(); }

void Backends::CudaBackend::invalidateResult() const {
  _impl_->result.token.isValid = false;
  _impl_->result.status = {};
#ifdef GUNDAM_USING_CUDA
  if( _impl_->context ){ _impl_->context->invalidate(); }
#endif
}

Backends::PropagationToken Backends::CudaBackend::requestPropagation(const PropagationInputs& inputs_) {
  // Every request invalidates earlier tokens, including requests that fail input
  // validation. Successful requests retain the device response cache.
  _impl_->result.token.isValid = false;
  _impl_->result.status = {};
  _impl_->timing = _impl_->buildTiming;
  const auto start = Clock::now();
  try{
    const auto& view = getEngineView();
    inputs_.validate(view.propagation);
    for( double value : inputs_.hostVariables ){
      if( not std::isfinite(value) ){ throw std::runtime_error("CudaBackend: host observables must be finite."); }
    }
    for( double value : inputs_.parameters.values ){
      if( not std::isfinite(value) ){ throw std::runtime_error("CudaBackend: parameter values must be finite."); }
    }
#ifdef GUNDAM_USING_CUDA
    _impl_->context->propagate(inputs_,
        _propagationMode_ == CudaPropagationMode::recomputeResponses, _reusePrediction_, _impl_->timing);
#else
    initializeDevice();
#endif
    auto& result = _impl_->result;
    result.status.backend = BackendStatus::Ready;
    result.status.eventWeights = OutputState::ReadyOnDevice;
    result.status.histograms = OutputState::ReadyOnDevice;
    result.status.eventState = _impl_->hasEventState ? OutputState::ReadyOnDevice : OutputState::NotRequested;
    result.status.sampleLikelihoods = OutputState::Failed;
    result.status.statLikelihood = OutputState::Failed;
    if( not view.likelihood.empty() ){
      if( hasPortableLikelihood(view.likelihood) ){
#ifdef GUNDAM_USING_CUDA
        result.likelihood = _impl_->context->readLikelihood();
        _impl_->timing.likelihoodReadbackBytes = sizeof(double);
#endif
      }
      else{
        readHistograms();
        const auto likelihoodStart = Clock::now();
        result.likelihood = Semantics::calculateLikelihood(view.likelihood, result.histSums, result.histSumSquares);
        _impl_->timing.likelihoodHostSeconds = elapsed(likelihoodStart);
      }
      result.status.statLikelihood = OutputState::ReadyOnHost;
    }
    if( _nextTokenId_ == 0 ){ throw std::runtime_error("CudaBackend token sequence exhausted."); }
    result.token = {_nextTokenId_++, true};
    _impl_->timing.uniqueDialCount = view.propagation.dials.size();
    _impl_->timing.cachedDialCount = _propagationMode_ == CudaPropagationMode::cachedResponses ? view.propagation.dials.size() : 0;
    _impl_->timing.eventDialIndexCount = view.propagation.eventDialIndices.size();
    _impl_->timing.splineScalarCount = view.propagation.dialPayloads.size();
    auto& totals = _impl_->totals;
    ++totals.propagationCount;
    _impl_->timing.propagationWallSeconds = elapsed(start);
    totals.propagationSeconds += _impl_->timing.propagationWallSeconds;
    totals.cachedDialSeconds += _impl_->timing.cachedDialStageSeconds;
    totals.eventWeightsSeconds += _impl_->timing.eventWeightsStageSeconds;
    totals.histogramSeconds += _impl_->timing.histogramStageSeconds;
    totals.likelihoodSeconds += _impl_->timing.likelihoodHostSeconds + _impl_->timing.likelihoodDeviceSeconds;
    return result.token;
  }
  catch(...){ invalidateResult(); throw; }
}

bool Backends::CudaBackend::isCurrentToken(const PropagationToken& token_) const {
  return token_.isValid and _impl_->result.token.isValid and token_.id == _impl_->result.token.id;
}

void Backends::CudaBackend::requireToken(const PropagationToken& token_) const {
  if( not isCurrentToken(token_) ){ throw std::runtime_error("Invalid CudaBackend propagation token."); }
}

Backends::PropagationStatus Backends::CudaBackend::getStatus(const PropagationToken& token_) const {
  return isCurrentToken(token_) ? _impl_->result.status : PropagationStatus{};
}

bool Backends::CudaBackend::isReady(const PropagationToken& token_) const {
  return isCurrentToken(token_) and _impl_->result.status.backend == BackendStatus::Ready;
}

// requestPropagation synchronizes its private stream; no work remains pending.
void Backends::CudaBackend::wait(const PropagationToken& token_) { requireToken(token_); }

void Backends::CudaBackend::readWeights() const {
  if( _impl_->result.status.eventWeights == OutputState::ReadyOnHost ){ return; }
#ifdef GUNDAM_USING_CUDA
  const auto start = Clock::now();
  try{ _impl_->context->readWeights(_impl_->result.eventWeights); }
  catch(...){ invalidateResult(); throw; }
  _impl_->timing.eventWeightReadbackSeconds += elapsed(start);
  _impl_->timing.eventWeightReadbackBytes += _impl_->result.eventWeights.size() * sizeof(double);
  _impl_->result.status.eventWeights = OutputState::ReadyOnHost;
#else
  throw std::runtime_error("CUDA support not compiled.");
#endif
}

void Backends::CudaBackend::readHistograms() const {
  if( _impl_->result.status.histograms == OutputState::ReadyOnHost ){ return; }
#ifdef GUNDAM_USING_CUDA
  const auto start = Clock::now();
  try{ _impl_->context->readHistograms(_impl_->result.histSums, _impl_->result.histSumSquares,
                                         _impl_->result.outsideSums, _impl_->result.outsideSumSquares); }
  catch(...){ invalidateResult(); throw; }
  _impl_->timing.histogramReadbackSeconds += elapsed(start);
  _impl_->timing.histogramReadbackBytes += 2 * (_impl_->result.histSums.size()
      + (_impl_->hasSmoothedBinning ? _impl_->result.outsideSums.size() : 0)) * sizeof(double);
  _impl_->result.status.histograms = OutputState::ReadyOnHost;
#else
  throw std::runtime_error("CUDA support not compiled.");
#endif
}

void Backends::CudaBackend::readEventState() const {
  if( not _impl_->hasEventState ){ throw std::runtime_error("CudaBackend: no observable state in this model."); }
  if( _impl_->result.status.eventState == OutputState::ReadyOnHost ){ return; }
#ifdef GUNDAM_USING_CUDA
  const auto start = Clock::now();
  try{ _impl_->timing.eventStateReadbackBytes += _impl_->context->readEventState(_impl_->eventState); }
  catch(...){ invalidateResult(); throw; }
  _impl_->timing.eventStateReadbackSeconds += elapsed(start);
  _impl_->result.status.eventState = OutputState::ReadyOnHost;
#else
  throw std::runtime_error("CUDA support not compiled.");
#endif
}

void Backends::CudaBackend::materialize(const PropagationToken& token_, OutputRequest output_) {
  requireToken(token_);
  if( output_ == OutputRequest::EventState ){ readEventState(); }
  else if( output_ == OutputRequest::EventWeights ){ readWeights(); }
  else if( output_ == OutputRequest::Histograms ){ readHistograms(); }
  else if( output_ == OutputRequest::StatLikelihood ){ (void)getLikelihood(token_); }
  else{ throw std::runtime_error("CudaBackend: requested output is not supported."); }
}

double Backends::CudaBackend::getLikelihood(const PropagationToken& token_) const {
  requireToken(token_);
  if( _impl_->result.status.statLikelihood != OutputState::ReadyOnHost ){
    throw std::runtime_error("CudaBackend: likelihood is not available for this model.");
  }
  return _impl_->result.likelihood;
}

const std::vector<double>& Backends::CudaBackend::getEventWeightsHostView(const PropagationToken& token_) const {
  requireToken(token_); readWeights(); return _impl_->result.eventWeights;
}
const std::vector<double>& Backends::CudaBackend::getHistogramSumsHostView(const PropagationToken& token_) const {
  requireToken(token_); readHistograms(); return _impl_->result.histSums;
}
const std::vector<double>& Backends::CudaBackend::getHistogramSumSquaresHostView(const PropagationToken& token_) const {
  requireToken(token_); readHistograms(); return _impl_->result.histSumSquares;
}
const Backends::EventState& Backends::CudaBackend::getEventStateHostView(const PropagationToken& token_) const {
  requireToken(token_); readEventState(); return _impl_->eventState;
}
const std::vector<double>& Backends::CudaBackend::getOutsideSumsHostView(const PropagationToken& token_) const {
  requireToken(token_); readHistograms(); return _impl_->result.outsideSums;
}
const std::vector<double>& Backends::CudaBackend::getOutsideSumSquaresHostView(const PropagationToken& token_) const {
  requireToken(token_); readHistograms(); return _impl_->result.outsideSumSquares;
}
Backends::BackendDeviceView Backends::CudaBackend::getDeviceView(const PropagationToken& token_) const {
  requireToken(token_);
#ifdef GUNDAM_USING_CUDA
  return _impl_->context->getDeviceView();
#else
  return {};
#endif
}
Backends::BackendTimingSummary Backends::CudaBackend::getLastTimingSummary() const { return _impl_->timing; }
Backends::BackendTimingTotals Backends::CudaBackend::getTimingTotals() const { return _impl_->totals; }

Backends::BackendResourceUsage Backends::CudaBackend::getResourceUsage() const {
#ifdef GUNDAM_USING_CUDA
  if( _impl_->context ){ return _impl_->context->getResourceUsage(); }
#endif
  return {};
}
