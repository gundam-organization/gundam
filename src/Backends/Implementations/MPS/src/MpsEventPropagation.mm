#include "MpsEventPropagation.h"
#include "MpsBackendInternal.h"
#include "DialDependencyGroups.h"
#include "LikelihoodModel.h"

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {
const char* const kDiagnosticKernelNames[]{"evaluate_propagation_dials", "propagate_event_state", "rebin_events", "clear_contribution_counts",
    "count_bin_contributions", "index_histogram_chunks", "scatter_bin_contributions", "reduce_histogram_chunks", "finish_dynamic_histograms", "reduce_histogram_chunks_parallel", "evaluate_likelihood_partials", "finish_likelihood"};
}

Backends::MpsEventPropagation::MpsEventPropagation(id<MTLDevice> device_, id<MTLCommandQueue> queue_, id<MTLLibrary> library_)
    : _device_(device_), _queue_(queue_) {

  for( std::size_t i = 0; i < _pipelines_.size(); ++i ){
    _pipelines_[i] = makePipeline(_device_, library_, [NSString stringWithUTF8String:kDiagnosticKernelNames[i]]);
    if( _pipelines_[i] == nil ){
      for( auto pipeline : _pipelines_ ){ [pipeline release]; }
      throw std::runtime_error("Could not create Metal event propagation pipelines.");
    }
  }
}

Backends::MpsEventPropagation::~MpsEventPropagation() {
  clearBuffers();
  for( auto pipeline : _pipelines_ ){ [pipeline release]; }
}

void Backends::MpsEventPropagation::clearBuffers() {
  for( auto& value : _buffers_ ){ releaseBuffer(value); }
}

void Backends::MpsEventPropagation::clear() {
  clearBuffers();
  _model_ = nullptr;
  _packed_ = DevicePropagationModel();
  _state_ = EventState();
  _cache_ = EventStateCache();
  _valid_ = false;
}

void Backends::MpsEventPropagation::build(const PropagationView& model_, const LikelihoodView& likelihood_) {
  clearBuffers();
  _model_ = &model_;
  _packed_.build(model_);
  _cache_.build(model_);
  _state_ = EventState();
  _materializedRevision_ = 0;
  _valid_ = false;
  _externalGenerations_.assign(model_.externalWeightBlocks.size(), 0);
  auto allocate = [&](Buffer slot_, std::size_t bytes_, bool shared_ = false){
    bytes_ = std::max(std::size_t(4), bytes_);
    LogThrowIf(bytes_ > _device_.maxBufferLength, "Event propagation allocation exceeds Metal maxBufferLength: " << bytes_);
    auto value = shared_ ? makeSharedEmptyBuffer(_device_, bytes_) : makePrivateEmptyBuffer(_device_, bytes_);
    LogThrowIf(value == nil, "Metal event propagation buffer allocation failed: " << bytes_);
    _buffers_[std::size_t(slot_)] = value;
  };
  auto upload = [&](Buffer slot_, const auto& values_){
    using Value = typename std::decay_t<decltype(values_)>::value_type;
    if( values_.empty() ){ allocate(slot_, sizeof(Value)); return; }
    LogThrowIf(values_.size() > _device_.maxBufferLength / sizeof(Value), "Event propagation upload exceeds Metal maxBufferLength.");
    _buffers_[std::size_t(slot_)] = makePrivateBuffer(_device_, _queue_, values_);
    LogThrowIf(buffer(slot_) == nil, "Metal event propagation upload failed.");
  };
  upload(Buffer::Events, _packed_.events);
  upload(Buffer::Samples, _packed_.samples);
  upload(Buffer::Dials, _packed_.dials);
  upload(Buffer::DialInputs, _packed_.dialInputs);
  upload(Buffer::DialInputRanges, _packed_.dialInputRanges);
  upload(Buffer::Targets, _packed_.variableResponses);
  upload(Buffer::Intervals, _packed_.intervals);
  upload(Buffer::Bins, _packed_.bins);
  upload(Buffer::Nodes, _packed_.nodes);
  _deviceLikelihood_ = hasPortableLikelihood(likelihood_);
  const auto likelihoodBins = packLikelihood<ObservableScalar>(likelihood_, model_.totalBins);
  _likelihoodBinCount_ = likelihoodBins.size();
  upload(Buffer::LikelihoodBins, likelihoodBins);
  upload(Buffer::LikelihoodCount, std::vector<DeviceIndex>{DeviceIndex(_likelihoodBinCount_)});
  allocate(Buffer::LikelihoodPartials, ((_likelihoodBinCount_ + 63) / 64) * sizeof(ObservableScalar));
  allocate(Buffer::LikelihoodTotal, sizeof(ObservableScalar));
  allocate(Buffer::LikelihoodReadback, sizeof(ObservableScalar), true);
  upload(Buffer::Bounds, _packed_.bounds);
  upload(Buffer::NodeBins, _packed_.nodeBins);
  upload(Buffer::ReusableBins, _packed_.reusableBins);
  upload(Buffer::WeightDialIndices, _packed_.weightDialIndices);
  upload(Buffer::VariableDialIndices, _packed_.variableDialIndices);
  upload(Buffer::Payloads, _packed_.payloads);
  upload(Buffer::Variables, _packed_.variables);
  upload(Buffer::FormulaPrograms, _packed_.formulaPrograms);
  upload(Buffer::FormulaInstructions, _packed_.formulaInstructions);
  upload(Buffer::FormulaConstants, _packed_.formulaConstants);
  upload(Buffer::ProgramInputs, _packed_.programInputs);
  upload(Buffer::CompositePrograms, _packed_.compositePrograms);
  upload(Buffer::CompositeInstructions, _packed_.compositeInstructions);
  allocate(Buffer::ProgramScratch, _packed_.programScratchSize * sizeof(ObservableScalar));
  allocate(Buffer::Parameters, std::max(std::size_t(1), model_.parameterCount) * sizeof(ObservableScalar), true);
  std::size_t externalCount{0};
  for( const auto& block : model_.externalWeightBlocks ){ externalCount = std::max(externalCount, block.offset + block.count); }
  allocate(Buffer::ExternalWeights, std::max(std::size_t(1), externalCount) * sizeof(ObservableScalar), true);
  allocate(Buffer::Responses, std::max(std::size_t(1), _packed_.dials.size()) * sizeof(ObservableScalar));
  allocate(Buffer::HostValues, std::max(std::size_t(1), _packed_.variables.size()) * sizeof(ObservableScalar), true);
  allocate(Buffer::HostEdits, _packed_.variables.size() * sizeof(DeviceIndex), true);
  allocate(Buffer::ChangedSamples, model_.samples.size() * sizeof(DeviceIndex), true);
  allocate(Buffer::Control, sizeof(DevicePropagationControl), true);
  for( auto slot : {Buffer::DirtyEvents, Buffer::HardBins, Buffer::EventCounts} ){ allocate(slot, model_.events.size() * sizeof(DeviceIndex)); }
  allocate(Buffer::Contributions, std::max(std::size_t(1), std::size_t(_packed_.contributionCapacity)) * sizeof(DeviceBinContribution));
  allocate(Buffer::EventWeights, model_.events.size() * sizeof(ObservableScalar));
  for( auto slot : {Buffer::BinCounts, Buffer::BinPositions, Buffer::ChunkOffsets} ){
    allocate(slot, (std::size_t(_packed_.histogramBins) + 1) * sizeof(DeviceIndex));
  }
  allocate(Buffer::BinContributions, std::size_t(_packed_.contributionCapacity) * sizeof(DeviceIndex));
  allocate(Buffer::Chunks, std::size_t(_packed_.histogramChunkCapacity) * sizeof(DeviceHistogramChunk));
  for( auto slot : {Buffer::PartialSums, Buffer::PartialSquares} ){
    allocate(slot, std::size_t(_packed_.histogramChunkCapacity) * sizeof(ObservableScalar));
  }
  for( auto slot : {Buffer::HistogramSums, Buffer::HistogramSquares} ){
    allocate(slot, std::size_t(_packed_.histogramBins) * sizeof(ObservableScalar));
  }
  // Readback buffers are created lazily for EventState/EventWeights, which fits
  // usually leave resident. Histogram readback is allocated lazily for diagnostics.
  std::memset(buffer(Buffer::Control).contents, 0, sizeof(DevicePropagationControl));
  buildDialCache();
  // Uploads have completed. Retain only the host metadata used at runtime.
  auto releaseHostStorage = [](auto& values_){ std::decay_t<decltype(values_)>().swap(values_); };
  releaseHostStorage(_packed_.dialInputs);
  releaseHostStorage(_packed_.dialInputRanges);
  releaseHostStorage(_packed_.variableResponses);
  releaseHostStorage(_packed_.intervals);
  releaseHostStorage(_packed_.bins);
  releaseHostStorage(_packed_.nodes);
  releaseHostStorage(_packed_.bounds);
  releaseHostStorage(_packed_.nodeBins);
  releaseHostStorage(_packed_.weightDialIndices);
  releaseHostStorage(_packed_.variableDialIndices);
  releaseHostStorage(_packed_.variables);
  releaseHostStorage(_packed_.payloads);
  releaseHostStorage(_packed_.formulaInstructions);
  releaseHostStorage(_packed_.formulaConstants);
  releaseHostStorage(_packed_.programInputs);
  releaseHostStorage(_packed_.compositePrograms);
  releaseHostStorage(_packed_.compositeInstructions);
}

void Backends::MpsEventPropagation::buildDialCache() {
  const auto& model = *_model_;
  _cachedParameters_.clear();
  _cachedExternalGenerations_.clear();
  _dialGroups_.clear();
  _multiDialGroups_.clear();
  _weightParameters_.clear();
  _weightExternalBlocks_.clear();
  auto groups = buildDialDependencyGroups<DeviceIndex>(model, model.dials);
  std::vector<DeviceIndex> indices = std::move(groups.parameterDialIndices);
  for( std::size_t parameter = 0; parameter < model.parameterCount; ++parameter ){
    const auto first = groups.parameterOffsets[parameter];
    const auto count = groups.parameterOffsets[parameter + 1] - first;
    if( count == 0 ){ continue; }
    _dialGroups_.push_back({parameter, first, count, false});
  }
  for( const auto& group : groups.externalDials ){
    _dialGroups_.push_back({group.first, indices.size(), group.second.size(), true});
    indices.insert(indices.end(), group.second.begin(), group.second.end());
  }
  for( const auto& group : groups.multiParameterDials ){
    _multiDialGroups_.push_back({group.first, indices.size(), group.second.size()});
    indices.insert(indices.end(), group.second.begin(), group.second.end());
  }
  _buffers_[std::size_t(Buffer::DialIndices)] = indices.empty() ? makePrivateEmptyBuffer(_device_, 4)
      : makePrivateBuffer(_device_, _queue_, indices);
  _buffers_[std::size_t(Buffer::DialRanges)] = makeSharedEmptyBuffer(_device_,
      std::max(std::size_t(1), _dialGroups_.size() + _multiDialGroups_.size()) * sizeof(DevicePropagationDialRange));
  LogThrowIf(not buffer(Buffer::DialIndices) or not buffer(Buffer::DialRanges), "Metal dial cache allocation failed.");
  std::vector<bool> used(model.dials.size(), false);
  for( auto index : model.eventDialIndices ){ used.at(index) = true; }
  for( std::size_t i = 0; i < used.size(); ++i ){
    if( not used[i] ){ continue; }
    const auto& dial = model.dials[i];
    if( dial.type == BackendDialType::Shift ){ continue; }
    visitDialDependencies(model, dial,
        [this](const auto& input_){ _weightParameters_.push_back(input_.parameterIndex); },
        [this](std::size_t block_){ _weightExternalBlocks_.push_back(block_); });
  }
  auto unique = [](auto& values_){
    std::sort(values_.begin(), values_.end());
    values_.erase(std::unique(values_.begin(), values_.end()), values_.end());
  };
  unique(_weightParameters_);
  unique(_weightExternalBlocks_);
}

std::size_t Backends::MpsEventPropagation::prepareDialCache(const PropagationInputs& inputs_, bool& updateWeights_) {
  auto& control = *static_cast<DevicePropagationControl*>(buffer(Buffer::Control).contents);
  control.dialRangeCount = 0;
  updateWeights_ = true;
  if( not _valid_ or not _enableDialCache_ ){ return _packed_.dials.size(); }
  auto parameterChanged = [&](std::size_t i_){ return _cachedParameters_[i_] != inputs_.parameters.values[i_]; };
  auto externalChanged = [&](std::size_t i_){ return _cachedExternalGenerations_[i_] != inputs_.externalWeights[i_].generation; };
  updateWeights_ = std::any_of(_weightParameters_.begin(), _weightParameters_.end(), parameterChanged)
      or std::any_of(_weightExternalBlocks_.begin(), _weightExternalBlocks_.end(), externalChanged);
  auto* ranges = static_cast<DevicePropagationDialRange*>(buffer(Buffer::DialRanges).contents);
  std::size_t count{0};
  for( const auto& group : _dialGroups_ ){
    if( not (group.external ? externalChanged(group.dependency) : parameterChanged(group.dependency)) ){ continue; }
    ranges[control.dialRangeCount++] = {DeviceIndex(count), DeviceIndex(group.first), DeviceIndex(group.count)};
    count += group.count;
  }
  for( const auto& group : _multiDialGroups_ ){
    if( not std::any_of(group.parameters.begin(), group.parameters.end(), parameterChanged) ){ continue; }
    ranges[control.dialRangeCount++] = {DeviceIndex(count), DeviceIndex(group.first), DeviceIndex(group.count)};
    count += group.count;
  }
  // Dense updates retain the direct full scan, avoiding a range lookup per dial.
  if( count > _packed_.dials.size() / 2 ){
    control.dialRangeCount = 0;
    return _packed_.dials.size();
  }
  return count;
}

void Backends::MpsEventPropagation::dispatch(id<MTLComputeCommandEncoder> encoder_, Kernel kernel_, std::size_t count_,
                                            std::initializer_list<Buffer> buffers_) {
  if( count_ == 0 ){ return; }
  // Normal dispatches share one command buffer. Only diagnostic passes submit
  // and wait once per kernel, making their GPU timestamps separately readable.
  id<MTLCommandBuffer> probeCommand = nil;
  if( _enableDiagnosticStageProfiling_ ){
    probeCommand = [_queue_ commandBuffer];
    encoder_ = [probeCommand computeCommandEncoder];
  }
  auto pipeline = _pipelines_[std::size_t(kernel_)];
  [encoder_ setComputePipelineState:pipeline];
  NSUInteger index{0};
  for( auto slot : buffers_ ){ [encoder_ setBuffer:buffer(slot) offset:0 atIndex:index++]; }
  NSUInteger width = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
  [encoder_ dispatchThreads:MTLSizeMake(count_, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  if( probeCommand != nil ){
    [encoder_ endEncoding];
    [probeCommand commit];
    [probeCommand waitUntilCompleted];
    LogThrowIf(probeCommand.status != MTLCommandBufferStatusCompleted, "MPS diagnostic kernel failed.");
    _diagnosticStageGpuSeconds_[kDiagnosticKernelNames[std::size_t(kernel_)]] += probeCommand.GPUEndTime - probeCommand.GPUStartTime;
  }
}

void Backends::MpsEventPropagation::copy(id<MTLBlitCommandEncoder> encoder_, Buffer source_, Buffer destination_, std::size_t bytes_) {
  if( not bytes_ ){ return; }
  if( buffer(destination_) == nil ){
    _buffers_[std::size_t(destination_)] = makeSharedEmptyBuffer(_device_, bytes_);
    LogThrowIf(buffer(destination_) == nil, "Metal event propagation readback allocation failed.");
  }
  [encoder_ copyFromBuffer:buffer(source_) sourceOffset:0 toBuffer:buffer(destination_) destinationOffset:0 size:bytes_];
}

void Backends::MpsEventPropagation::propagate(const PropagationInputs& inputs_, PropagationResult& result_, BackendTimingSummary& timing_) {
  _diagnosticStageGpuSeconds_.clear();
  const auto& model = *_model_;
  inputs_.validate(model);
  _cache_.prepare(inputs_);
  uploadParameters<ObservableScalar>(buffer(Buffer::Parameters), inputs_.parameters, timing_);
  uploadExternalWeights<ObservableScalar>(buffer(Buffer::ExternalWeights), _externalGenerations_, inputs_, timing_);
  auto& control = *static_cast<DevicePropagationControl*>(buffer(Buffer::Control).contents);
  control.error = control.rebinnedEvents = control.candidates = control.editCount = 0;
  control.rebuildIndex = not _valid_;
  control.firstPropagation = not _valid_;
  bool updateWeights{true};
  auto dialCount = prepareDialCache(inputs_, updateWeights);
  control.updateWeights = updateWeights;
  _lastEvaluatedDials_ = dialCount;
  _lastUpdatedWeights_ = updateWeights;
  control.updateVariables = _cache_.variableInputsChanged();
  control.forceBinning = inputs_.forceBinningUpdate;
  control.eventCount = DeviceIndex(model.events.size());
  control.histogramBins = _packed_.histogramBins;
  control.dialCount = DeviceIndex(dialCount);
  control.sampleCount = DeviceIndex(model.samples.size());
  control.hasReweightCap = model.hasReweightCap;
  control.maxReweight = ObservableScalar(model.maxReweight);
  std::memset(buffer(Buffer::ChangedSamples).contents, 0, std::max(std::size_t(1), model.samples.size()) * sizeof(DeviceIndex));
  if( not inputs_.hostVariables.empty() ){
    auto* flags = static_cast<DeviceIndex*>(buffer(Buffer::HostEdits).contents);
    auto* values = static_cast<ObservableScalar*>(buffer(Buffer::HostValues).contents);
    std::fill(flags, flags + model.variableValues.size(), 0);
    for( std::size_t i = 0; i < inputs_.hostVariables.size(); ++i ){
      if( _cache_.hostVariableChanged(i, inputs_) ){
        flags[i] = 1;
        values[i] = ObservableScalar(inputs_.hostVariables[i]);
        ++control.editCount;
      }
    }
  }
  auto encodeStart = std::chrono::steady_clock::now();
  auto command = [_queue_ commandBuffer];
  auto encoder = [command computeCommandEncoder];
  dispatch(encoder, Kernel::Dials, dialCount, {Buffer::Dials, Buffer::Parameters, Buffer::Payloads, Buffer::ExternalWeights, Buffer::Responses, Buffer::Control, Buffer::DialIndices, Buffer::DialRanges, Buffer::DialInputs, Buffer::DialInputRanges,
      Buffer::FormulaPrograms, Buffer::FormulaInstructions, Buffer::FormulaConstants, Buffer::ProgramInputs,
      Buffer::CompositePrograms, Buffer::CompositeInstructions, Buffer::ProgramScratch});
  if( updateWeights or _cache_.needsUpdate(inputs_) or not _valid_ ){
    dispatch(encoder, Kernel::Events, model.events.size(), {Buffer::Events, Buffer::Samples, Buffer::Targets, Buffer::WeightDialIndices,
    Buffer::VariableDialIndices, Buffer::Responses, Buffer::HostValues, Buffer::HostEdits, Buffer::Variables, Buffer::DirtyEvents, Buffer::EventWeights, Buffer::Control});
  }
  if( _cache_.needsUpdate(inputs_) || not _valid_ ){
    dispatch(encoder, Kernel::Binning, model.events.size(), {Buffer::Events, Buffer::Samples, Buffer::Nodes, Buffer::Bounds, Buffer::NodeBins,
      Buffer::Bins, Buffer::Intervals, Buffer::Variables, Buffer::DirtyEvents, Buffer::HardBins, Buffer::EventCounts, Buffer::Contributions, Buffer::ChangedSamples, Buffer::Control, Buffer::ReusableBins});
    dispatch(encoder, Kernel::ClearCounts, _packed_.histogramBins, {Buffer::BinCounts, Buffer::Control});
    dispatch(encoder, Kernel::CountContributions, model.events.size(), {Buffer::Events, Buffer::EventCounts, Buffer::Contributions, Buffer::BinCounts, Buffer::Control});
    dispatch(encoder, Kernel::IndexChunks, 1, {Buffer::BinCounts, Buffer::BinPositions, Buffer::ChunkOffsets, Buffer::Chunks, Buffer::Control});
    dispatch(encoder, Kernel::Scatter, model.events.size(), {Buffer::Events, Buffer::EventCounts, Buffer::Contributions, Buffer::BinPositions, Buffer::BinContributions, Buffer::Control});
  }
  const bool parallel = _pipelines_[std::size_t(Kernel::ReduceParallel)].threadExecutionWidth == 32;
  dispatch(encoder, parallel ? Kernel::ReduceParallel : Kernel::Reduce, std::size_t(_packed_.histogramChunkCapacity) * (parallel ? 32 : 1), {Buffer::Chunks, Buffer::BinContributions, Buffer::Contributions,
    Buffer::EventWeights, Buffer::PartialSums, Buffer::PartialSquares, Buffer::Control});
  dispatch(encoder, Kernel::Finish, _packed_.histogramBins, {Buffer::ChunkOffsets, Buffer::PartialSums, Buffer::PartialSquares,
    Buffer::HistogramSums, Buffer::HistogramSquares, Buffer::Control});
  if( _deviceLikelihood_ ){
    dispatch(encoder, Kernel::LikelihoodPartials, (_likelihoodBinCount_ + 63) / 64,
        {Buffer::LikelihoodBins, Buffer::HistogramSums, Buffer::HistogramSquares, Buffer::LikelihoodPartials, Buffer::LikelihoodCount});
    dispatch(encoder, Kernel::LikelihoodFinish, 1, {Buffer::LikelihoodPartials, Buffer::LikelihoodTotal, Buffer::LikelihoodCount});
  }
  [encoder endEncoding];
  if( _deviceLikelihood_ ){
    auto blit = [command blitCommandEncoder];
    copy(blit, Buffer::LikelihoodTotal, Buffer::LikelihoodReadback, sizeof(ObservableScalar));
    [blit endEncoding];
  }
  timing_.commandEncodeSeconds += secondsSince(encodeStart);
  auto waitStart = std::chrono::steady_clock::now();
  [command commit];
  [command waitUntilCompleted];
  timing_.deviceWaitSeconds += secondsSince(waitStart);
  timing_.devicePropagationSeconds += command.GPUEndTime - command.GPUStartTime;
  for( const auto& stage : _diagnosticStageGpuSeconds_ ){ timing_.devicePropagationSeconds += stage.second; }
  if( command.status != MTLCommandBufferStatusCompleted || control.error ){
    _cache_.invalidate();
    _valid_ = false;
    throw std::runtime_error("Metal event propagation failed (nonfinite observable, invalid bin fractions or capacity overflow), error " + std::to_string(control.error));
  }
  _cachedParameters_ = inputs_.parameters.values;
  _cachedExternalGenerations_.resize(inputs_.externalWeights.size());
  for( std::size_t i = 0; i < inputs_.externalWeights.size(); ++i ){ _cachedExternalGenerations_[i] = inputs_.externalWeights[i].generation; }
  _lastRebuiltIndex_ = control.rebuildIndex != 0;
  _valid_ = true;
  if( _deviceLikelihood_ ){
    const auto value = *static_cast<const ObservableScalar*>(buffer(Buffer::LikelihoodReadback).contents);
    result_.likelihood = double(value.high) + value.low;
    if( std::isnan(result_.likelihood) ){
      invalidate();
      throw std::runtime_error("MPS: invalid statistical likelihood.");
    }
    timing_.likelihoodReadbackBytes += sizeof(ObservableScalar);
  }
  else{ materializeHistograms(result_, timing_); }
  for( const auto& stage : _diagnosticStageGpuSeconds_ ){
    if( stage.first == "evaluate_likelihood_partials" || stage.first == "finish_likelihood" ){
      timing_.likelihoodDeviceSeconds += stage.second;
    }
  }
  timing_.rebinnedEventCount = control.rebinnedEvents;
  timing_.binCandidateCount = control.candidates;
  timing_.binContributionCount = control.contributionCount;
  timing_.uniqueDialCount = model.dials.size();
  timing_.eventDialIndexCount = model.eventDialIndices.size() + model.variableDialIndices.size();
  timing_.contributionBufferBytes = std::size_t(_packed_.contributionCapacity) * sizeof(DeviceBinContribution);
  std::vector<std::uint8_t> changed(model.samples.size());
  const auto* samples = static_cast<const DeviceIndex*>(buffer(Buffer::ChangedSamples).contents);
  for( std::size_t i = 0; i < changed.size(); ++i ){ changed[i] = samples[i] != 0; }
  _cache_.commit(inputs_, changed);
}

void Backends::MpsEventPropagation::materializeHistograms(PropagationResult& result_, BackendTimingSummary& timing_) {
  if( result_.histSums.size() == std::size_t(_model_->totalBins) ){ return; }
  auto start = std::chrono::steady_clock::now();
  auto command = [_queue_ commandBuffer];
  auto blit = [command blitCommandEncoder];
  const auto bytes = std::size_t(_packed_.histogramBins) * sizeof(ObservableScalar);
  copy(blit, Buffer::HistogramSums, Buffer::ReadbackSums, bytes);
  copy(blit, Buffer::HistogramSquares, Buffer::ReadbackSquares, bytes);
  [blit endEncoding];
  [command commit];
  [command waitUntilCompleted];
  LogThrowIf(command.status != MTLCommandBufferStatusCompleted, "Metal histogram readback failed.");
  const auto* sums = static_cast<const ObservableScalar*>(buffer(Buffer::ReadbackSums).contents);
  const auto* squares = static_cast<const ObservableScalar*>(buffer(Buffer::ReadbackSquares).contents);
  auto readValues = [](const ObservableScalar* values_, std::size_t count_, std::vector<double>& output_){
    output_.resize(count_);
    for( std::size_t i = 0; i < count_; ++i ){ output_[i] = double(values_[i].high) + values_[i].low; }
  };
  readValues(sums, _model_->totalBins, result_.histSums);
  readValues(squares, _model_->totalBins, result_.histSumSquares);
  readValues(sums + _model_->totalBins, _packed_.histogramBins - _model_->totalBins, result_.outsideSums);
  readValues(squares + _model_->totalBins, _packed_.histogramBins - _model_->totalBins, result_.outsideSumSquares);
  timing_.histogramReadbackBytes += 2 * bytes;
  timing_.histogramReadbackSeconds += secondsSince(start);
}

void Backends::MpsEventPropagation::materializeWeights(PropagationResult& result_, BackendTimingSummary& timing_) {
  if( result_.eventWeights.size() == _model_->events.size() ){ return; }
  auto start = std::chrono::steady_clock::now();
  auto command = [_queue_ commandBuffer];
  auto encoder = [command blitCommandEncoder];
  std::size_t bytes = _model_->events.size() * sizeof(ObservableScalar);
  copy(encoder, Buffer::EventWeights, Buffer::ReadbackWeights, bytes);
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  LogThrowIf(command.status != MTLCommandBufferStatusCompleted, "Metal event weight readback failed.");
  auto* values = static_cast<ObservableScalar*>(buffer(Buffer::ReadbackWeights).contents);
  result_.eventWeights.resize(_model_->events.size());
  for( std::size_t i = 0; i < result_.eventWeights.size(); ++i ){
    result_.eventWeights[i] = double(values[i].high) + values[i].low;
  }
  timing_.eventWeightReadbackBytes += bytes;
  timing_.eventWeightReadbackSeconds += secondsSince(start);
}

const Backends::EventState& Backends::MpsEventPropagation::materializeState(BackendTimingSummary& timing_) {
  EventState revisions;
  _cache_.publishRevisions(revisions);
  if( _materializedRevision_ == revisions.revision ){ return _state_; }
  auto start = std::chrono::steady_clock::now();
  auto command = [_queue_ commandBuffer];
  auto encoder = [command blitCommandEncoder];
  const auto& model = *_model_;
  std::size_t variableBytes = model.variableValues.size() * sizeof(ObservableScalar);
  std::size_t eventBytes = model.events.size() * sizeof(DeviceIndex);
  std::size_t contributionBytes = std::size_t(_packed_.contributionCapacity) * sizeof(DeviceBinContribution);
  copy(encoder, Buffer::Variables, Buffer::ReadbackVariables, variableBytes);
  copy(encoder, Buffer::HardBins, Buffer::ReadbackBins, eventBytes);
  copy(encoder, Buffer::EventCounts, Buffer::ReadbackCounts, eventBytes);
  copy(encoder, Buffer::Contributions, Buffer::ReadbackContributions, contributionBytes);
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  LogThrowIf(command.status != MTLCommandBufferStatusCompleted, "Metal event state readback failed.");
  _state_ = std::move(revisions);
  if( variableBytes ){
    auto* values = static_cast<ObservableScalar*>(buffer(Buffer::ReadbackVariables).contents);
    _state_.variables.resize(model.variableValues.size());
    for( std::size_t i = 0; i < _state_.variables.size(); ++i ){ _state_.variables[i] = double(values[i].high) + values[i].low; }
  }
  _state_.bins.resize(model.events.size());
  _state_.contributions.resize(model.events.size());
  const auto* bins = static_cast<int*>(buffer(Buffer::ReadbackBins).contents);
  const auto* counts = static_cast<DeviceIndex*>(buffer(Buffer::ReadbackCounts).contents);
  const auto* contributions = static_cast<DeviceBinContribution*>(buffer(Buffer::ReadbackContributions).contents);
  for( std::size_t i = 0; i < model.events.size(); ++i ){
    _state_.bins[i] = bins[i];
    const auto& event = _packed_.events[i];
    const auto& sample = _packed_.samples[event.sample];
    for( std::size_t j = 0; j < counts[i]; ++j ){
      const auto& contribution = contributions[event.firstContribution + j];
      int bin = contribution.binIndex >= model.totalBins ? -1 : contribution.binIndex - int(sample.binOffset);
      _state_.contributions[i].push_back({bin, double(contribution.fraction.high) + contribution.fraction.low});
    }
  }
  _materializedRevision_ = _state_.revision;
  timing_.eventStateReadbackBytes += variableBytes + 2 * eventBytes + contributionBytes;
  timing_.eventStateReadbackSeconds += secondsSince(start);
  return _state_;
}

Backends::BackendDeviceView Backends::MpsEventPropagation::getDeviceView() const {
  BackendDeviceView view;
  view.device = _device_;
  view.eventWeights = buffer(Buffer::EventWeights);
  view.eventWeightsBytes = _model_->events.size() * sizeof(ObservableScalar);
  view.histSums = buffer(Buffer::HistogramSums);
  view.histSumSquares = buffer(Buffer::HistogramSquares);
  view.histogramBytes = std::size_t(_model_->totalBins) * sizeof(ObservableScalar);
  return view;
}

std::string Backends::MpsEventPropagation::getExecutionState() const {
  return JsonType{{"evaluatedDials", _lastEvaluatedDials_}, {"updatedEventWeights", _lastUpdatedWeights_},
                  {"rebuiltContributionIndex", _lastRebuiltIndex_}}.dump();
}

std::string Backends::MpsEventPropagation::getDiagnostics() const {
  JsonType out;
  out["likelihoodExecution"] = _deviceLikelihood_ ? "device" : "host";
  out["evaluatedDials"] = _lastEvaluatedDials_;
  out["updatedEventWeights"] = _lastUpdatedWeights_;
  out["rebuiltContributionIndex"] = _lastRebuiltIndex_;
  out["dialCacheEnabled"] = _enableDialCache_;
  out["parallelHistogramsEnabled"] = _pipelines_[std::size_t(Kernel::ReduceParallel)].threadExecutionWidth == 32;
  out["lastStageGpuSeconds"] = _diagnosticStageGpuSeconds_;
  out["reusableBins"] = std::count_if(_packed_.reusableBins.begin(), _packed_.reusableBins.end(),
      [](DeviceIndex position_){ return position_ != DeviceIndex(-1); });
  out["contributionCapacity"] = _packed_.contributionCapacity;
  out["histogramChunkCapacity"] = _packed_.histogramChunkCapacity;
  out["buffers"] = JsonType::array();
  const char* names[]{"Events", "Samples", "Dials", "DialInputs", "DialInputRanges", "Targets", "Intervals", "Bins", "Nodes", "Bounds", "NodeBins", "ReusableBins", "WeightDialIndices", "VariableDialIndices", "Payloads", "Parameters", "ExternalWeights", "Responses", "DialIndices", "DialRanges", "Variables", "HostValues", "HostEdits", "DirtyEvents", "HardBins", "EventCounts", "Contributions", "ChangedSamples", "Control", "EventWeights", "BinCounts", "BinPositions", "BinContributions", "ChunkOffsets", "Chunks", "PartialSums", "PartialSquares", "HistogramSums", "HistogramSquares", "ReadbackSums", "ReadbackSquares", "ReadbackVariables", "ReadbackBins", "ReadbackCounts", "ReadbackContributions", "ReadbackWeights", "FormulaPrograms", "FormulaInstructions", "FormulaConstants", "ProgramInputs", "CompositePrograms", "CompositeInstructions", "ProgramScratch", "LikelihoodBins", "LikelihoodCount", "LikelihoodPartials", "LikelihoodTotal", "LikelihoodReadback"};
  static_assert(sizeof(names) / sizeof(*names) == std::size_t(Buffer::Count));
  for( std::size_t i = 0; i < _buffers_.size(); ++i ){
    auto value = _buffers_[i];
    if( value == nil ){ continue; }
    out["buffers"].push_back({{"name", names[i]}, {"bytes", value.length},
      {"storage", value.storageMode == MTLStorageModeShared ? "Shared" : "Private"}});
  }
  out["pipelines"] = JsonType::array();
  for( std::size_t i = 0; i < _pipelines_.size(); ++i ){
    auto pipeline = _pipelines_[i];
    out["pipelines"].push_back({{"name", kDiagnosticKernelNames[i]}, {"simdWidth", pipeline.threadExecutionWidth},
      {"maxThreadsPerThreadgroup", pipeline.maxTotalThreadsPerThreadgroup},
      {"dispatchThreadsPerThreadgroup", std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup)},
      {"staticThreadgroupMemoryBytes", pipeline.staticThreadgroupMemoryLength}});
  }
  return out.dump();
}
