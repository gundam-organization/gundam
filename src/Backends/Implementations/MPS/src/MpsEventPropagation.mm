#include "MpsEventPropagation.h"
#include "MpsBackendInternal.h"
#include "MpsGradientModel.h"
#include "BinFactors.h"
#include "DialDependencyGroups.h"
#include "LikelihoodModel.h"
#include "ProducerGradientPublications.h"

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {
const char* const kDiagnosticKernelNames[]{"evaluate_propagation_dials", "propagate_event_state", "rebin_events", "clear_contribution_counts",
    "count_bin_contributions", "index_histogram_chunks", "scatter_bin_contributions", "reduce_histogram_chunks", "finish_dynamic_histograms", "reduce_histogram_chunks_parallel", "evaluate_likelihood_partials", "finish_likelihood", "gradient_dials", "gradient_bins", "gradient_events", "gradient_reduce", "apply_bin_factors"};
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
  _dialCount_ = 0;
  _packed_ = DevicePropagationModel();
  _state_ = EventState();
  _cache_ = EventStateCache();
  _valid_ = false;
  _producerDials_.clear(); _producerInputCounts_.clear(); _producerGradients_.clear(); _producerGradientsValid_ = false;
}

void Backends::MpsEventPropagation::updateDataHistograms(const LikelihoodView& likelihood_) {
  LogThrowIf(_model_ == nullptr, "Metal data updates require a built model.");
  const auto bins = packLikelihood<ObservableScalar>(likelihood_, _model_->totalBins);
  LogThrowIf(bins.size() != _likelihoodWork_.binCount
             or hasPortableLikelihood(likelihood_) != _deviceLikelihood_,
             "Cannot change the Metal likelihood layout during a data update.");
  const bool updateGradientBins = hasGradientBuffers() and _gradientWork_.bins != 0;
  const auto gradientBins = updateGradientBins
      ? MpsGradientModel::packBins(bins, _model_->totalBins) : decltype(bins){};
  auto updated = bins.empty() ? nil : makePrivateBuffer(_device_, _queue_, bins);
  LogThrowIf(not bins.empty() and updated == nil, "Metal data histogram upload failed.");
  auto updatedGradient = updateGradientBins ? makePrivateBuffer(_device_, _queue_, gradientBins) : nil;
  if( updateGradientBins and updatedGradient == nil ){
    [updated release];
    throw std::runtime_error("Metal gradient data histogram upload failed.");
  }
  if( updated != nil ){
    releaseBuffer(_buffers_[std::size_t(Buffer::LikelihoodBins)]);
    _buffers_[std::size_t(Buffer::LikelihoodBins)] = updated;
  }
  if( updatedGradient != nil ){
    releaseBuffer(_buffers_[std::size_t(Buffer::GradientBins)]);
    _buffers_[std::size_t(Buffer::GradientBins)] = updatedGradient;
  }
  _valid_ = false; // Force a fresh likelihood even for identical MC parameters.
}

void Backends::MpsEventPropagation::build(const PropagationView& model_, const LikelihoodView& likelihood_) {
  clearBuffers();
  _model_ = &model_;
  _dialCount_ = model_.dials.size();
  _producerDials_.clear();
  _producerInputCounts_.clear();
  for( std::size_t i = 0; i < model_.dials.size(); ++i ){
    const auto& dial = model_.dials[i];
    if( dial.type == BackendDialType::ExternalWeight or isTableDial(dial.type) ){
      _producerDials_.push_back(i);
      _producerInputCounts_.push_back(dial.inputCount);
    }
  }
  _producerGradients_.assign(_producerDials_.size(), {});
  _producerGradientsValid_ = false;
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
  auto upload = [&](Buffer slot_, const char* name_, const auto& values_){
    using Value = typename std::decay_t<decltype(values_)>::value_type;
    if( values_.empty() ){ allocate(slot_, sizeof(Value)); return; }
    LogThrowIf(values_.size() > _device_.maxBufferLength / sizeof(Value), "Event propagation upload exceeds Metal maxBufferLength.");
    LogInfo << "Uploading " << name_ << " to Metal (" << values_.size() * sizeof(Value) / (1024. * 1024.)
            << " MiB, " << values_.size() << " entries)..." << std::endl;
    _buffers_[std::size_t(slot_)] = makePrivateBuffer(_device_, _queue_, values_);
    LogThrowIf(buffer(slot_) == nil, "Metal event propagation upload failed for " << name_);
  };
  auto uploadAndRelease = [&](Buffer slot_, const char* name_, auto& values_){
    upload(slot_, name_, values_);
    std::decay_t<decltype(values_)>().swap(values_);
  };
  LogInfo << "Packing and uploading the MPS event model by section..." << std::endl;
  using Section = DevicePropagationModel::Section;
  _packed_.build(model_, [&](Section section_){
    switch( section_ ){
      case Section::Dials:
        uploadAndRelease(Buffer::Dials, "dial descriptors", _packed_.dials);
        uploadAndRelease(Buffer::DialInputs, "dial inputs", _packed_.dialInputs);
        uploadAndRelease(Buffer::DialInputRanges, "dial input ranges", _packed_.dialInputRanges);
        uploadAndRelease(Buffer::FormulaPrograms, "formula programs", _packed_.formulaPrograms);
        uploadAndRelease(Buffer::FormulaInstructions, "formula instructions", _packed_.formulaInstructions);
        uploadAndRelease(Buffer::FormulaConstants, "formula constants", _packed_.formulaConstants);
        uploadAndRelease(Buffer::ProgramInputs, "program inputs", _packed_.programInputs);
        uploadAndRelease(Buffer::CompositePrograms, "composite programs", _packed_.compositePrograms);
        uploadAndRelease(Buffer::CompositeInstructions, "composite instructions", _packed_.compositeInstructions);
        break;
      case Section::Payloads:
        uploadAndRelease(Buffer::Payloads, "dial payloads", _packed_.payloads);
        break;
      case Section::Variables:
        uploadAndRelease(Buffer::Variables, "event variables", _packed_.variables);
        break;
      case Section::DialIndices:
        uploadAndRelease(Buffer::WeightDialIndices, "weight dial indices", _packed_.weightDialIndices);
        uploadAndRelease(Buffer::VariableDialIndices, "variable dial indices", _packed_.variableDialIndices);
        break;
      case Section::Targets:
        uploadAndRelease(Buffer::Targets, "variable responses", _packed_.variableResponses);
        break;
      case Section::Binning:
        upload(Buffer::Samples, "sample binning", _packed_.samples);
        uploadAndRelease(Buffer::Intervals, "binning intervals", _packed_.intervals);
        uploadAndRelease(Buffer::Bins, "bins", _packed_.bins);
        uploadAndRelease(Buffer::Nodes, "bin search nodes", _packed_.nodes);
        uploadAndRelease(Buffer::Bounds, "bin search bounds", _packed_.bounds);
        uploadAndRelease(Buffer::NodeBins, "bin search indices", _packed_.nodeBins);
        upload(Buffer::ReusableBins, "reusable bin indices", _packed_.reusableBins);
        break;
      case Section::Events:
        upload(Buffer::Events, "events", _packed_.events);
        break;
    }
  });
  _deviceLikelihood_ = hasPortableLikelihood(likelihood_);
  const auto likelihoodBins = packLikelihood<ObservableScalar>(likelihood_, model_.totalBins);
  LogThrowIf(likelihoodBins.size() > std::numeric_limits<DeviceIndex>::max(), "Metal likelihood bin count exceeds index range.");
  _likelihoodWork_.binCount = DeviceIndex(likelihoodBins.size());
  // Keep many lanes active for small likelihoods. Larger likelihoods amortize
  // per-bin dispatch traffic over short chunks; final reduction is parallel.
  _likelihoodWork_.binsPerThread = likelihoodBins.size() <= 8192 ? 1 : 8;
  _likelihoodWork_.partialCount = DeviceIndex((likelihoodBins.size() + _likelihoodWork_.binsPerThread - 1)
      / _likelihoodWork_.binsPerThread);
  _lastUpdatedPrediction_ = true;
  upload(Buffer::LikelihoodBins, "likelihood bins", likelihoodBins);
  upload(Buffer::LikelihoodCount, "likelihood work layout", std::vector<LikelihoodWorkLayout>{_likelihoodWork_});
  allocate(Buffer::LikelihoodPartials, _likelihoodWork_.partialCount * sizeof(ObservableScalar));
  allocate(Buffer::LikelihoodTotal, sizeof(ObservableScalar));
  allocate(Buffer::LikelihoodReadback, sizeof(ObservableScalar), true);
  allocate(Buffer::ProgramScratch, _packed_.programScratchSize * sizeof(ObservableScalar));
  allocate(Buffer::Parameters, std::max(std::size_t(1), model_.parameterCount) * sizeof(ObservableScalar), true);
  std::size_t externalCount{0};
  for( const auto& block : model_.externalWeightBlocks ){ externalCount = std::max(externalCount, block.offset + block.count); }
  allocate(Buffer::ExternalWeights, std::max(std::size_t(1), externalCount) * sizeof(ObservableScalar), true);
  allocate(Buffer::Responses, std::max(std::size_t(1), model_.dials.size()) * sizeof(ObservableScalar));
  allocate(Buffer::HostValues, std::max(std::size_t(1), model_.variableValues.size()) * sizeof(ObservableScalar), true);
  allocate(Buffer::HostEdits, model_.variableValues.size() * sizeof(DeviceIndex), true);
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
  std::size_t bufferBytes{0};
  for( auto value : _buffers_ ){ if( value != nil ){ bufferBytes += value.length; } }
  LogInfo << "MPS buffers ready: " << model_.events.size() << " events, "
          << model_.dials.size() << " dials, " << model_.totalBins << " bins; "
          << bufferBytes / (1024. * 1024.) << " MiB allocated (private and shared Metal buffers)." << std::endl;
}

std::vector<Backends::DeviceIndex> Backends::MpsEventPropagation::findBinFactors() const {
  return Backends::findBinFactors(*_model_, _packed_.histogramBins, [this](const auto& event_){
    const auto& packed = _packed_.events[event_.resultIndex];
    if( packed.fixedBin < 0 ){ return std::size_t(-1); }
    return std::size_t(_packed_.samples[packed.sample].binOffset) + packed.fixedBin;
  }, [this](DeviceIndex dial_){ return _model_->dials[dial_].type == BackendDialType::Norm; });
}

bool Backends::MpsEventPropagation::usesBinFactorization() const {
  return _enableBinFactorization_ and _enableDialCache_ and _factoredParameterCount_ != 0;
}

void Backends::MpsEventPropagation::buildDialCache() {
  const auto& model = *_model_;
  _cachedParameters_.clear();
  _cachedExternalGenerations_.clear();
  _dialGroups_.clear();
  _multiDialGroups_.clear();
  _weightParameters_.clear();
  _weightExternalBlocks_.clear();
  _residualWeightParameters_.assign(model.parameterCount, false);
  _lastReusedResidualHistograms_ = false;
  _binFactorReuseCount_ = 0;
  _eventWeightsValid_ = false;
  const auto binFactors = findBinFactors();
  std::vector<bool> factoredDials(model.dials.size(), false);
  _factoredBinCount_ = 0;
  for( auto dial : binFactors ){
    if( dial == noBinFactor ){ continue; }
    factoredDials[dial] = true;
    ++_factoredBinCount_;
  }
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
  if( not indices.empty() ){
    LogInfo << "Uploading dial cache indices to Metal (" << indices.size() * sizeof(DeviceIndex) / (1024. * 1024.)
            << " MiB, " << indices.size() << " entries)..." << std::endl;
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
        [&](const auto& input_){
          _weightParameters_.push_back(input_.parameterIndex);
          if( not factoredDials[i] ){ _residualWeightParameters_[input_.parameterIndex] = true; }
        },
        [this](std::size_t block_){ _weightExternalBlocks_.push_back(block_); });
  }
  auto unique = [](auto& values_){
    std::sort(values_.begin(), values_.end());
    values_.erase(std::unique(values_.begin(), values_.end()), values_.end());
  };
  unique(_weightParameters_);
  unique(_weightExternalBlocks_);
  _factoredParameterCount_ = std::count_if(_weightParameters_.begin(), _weightParameters_.end(),
      [&](std::size_t parameter_){ return not _residualWeightParameters_[parameter_]; });
  _buffers_[std::size_t(Buffer::BinFactors)] = _factoredParameterCount_ ? makePrivateBuffer(_device_, _queue_, binFactors)
      : makePrivateEmptyBuffer(_device_, sizeof(DeviceIndex));
  LogThrowIf(buffer(Buffer::BinFactors) == nil, "Metal bin factor allocation failed.");
  if( _factoredParameterCount_ ){
    for( auto slot : {Buffer::ResidualSums, Buffer::ResidualSquares} ){
      _buffers_[std::size_t(slot)] = makePrivateEmptyBuffer(_device_, binFactors.size() * sizeof(ObservableScalar));
      LogThrowIf(buffer(slot) == nil, "Metal residual histogram allocation failed.");
    }
    if( usesBinFactorization() ){
      LogInfo << "MPS bin factorization: " << _factoredBinCount_ << " fixed bins, "
              << _factoredParameterCount_ << " parameters can reuse cached histogram sums." << std::endl;
    }
  }
}

std::size_t Backends::MpsEventPropagation::prepareDialCache(const PropagationInputs& inputs_, bool& updateWeights_,
                                                         bool& updateResidualWeights_) {
  auto& control = *static_cast<DevicePropagationControl*>(buffer(Buffer::Control).contents);
  const auto dialCount = _dialCount_;
  control.dialRangeCount = 0;
  updateWeights_ = true;
  updateResidualWeights_ = true;
  if( not _valid_ or not _enableDialCache_ ){ return dialCount; }
  auto parameterChanged = [&](std::size_t i_){ return _cachedParameters_[i_] != inputs_.parameters.values[i_]; };
  auto externalChanged = [&](std::size_t i_){ return _cachedExternalGenerations_[i_] != inputs_.externalWeights[i_].generation; };
  updateResidualWeights_ = updateWeights_ = std::any_of(_weightExternalBlocks_.begin(), _weightExternalBlocks_.end(), externalChanged);
  for( auto parameter : _weightParameters_ ){
    if( not parameterChanged(parameter) ){ continue; }
    updateWeights_ = true;
    updateResidualWeights_ = updateResidualWeights_ or _residualWeightParameters_[parameter];
  }
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
  if( count > dialCount / 2 ){
    control.dialRangeCount = 0;
    return dialCount;
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
  NSUInteger width = (kernel_ == Kernel::LikelihoodFinish or kernel_ == Kernel::GradientReduce) ? GUNDAM_LIKELIHOOD_REDUCTION_WIDTH
      : std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
  LogThrowIf(width > pipeline.maxTotalThreadsPerThreadgroup, "Metal likelihood reduction threadgroup is unsupported.");
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

void Backends::MpsEventPropagation::encodeEventWeights(id<MTLComputeCommandEncoder> encoder_) {
  dispatch(encoder_, Kernel::Events, _model_->events.size(), {Buffer::Events, Buffer::Samples, Buffer::Targets, Buffer::WeightDialIndices,
      Buffer::VariableDialIndices, Buffer::Responses, Buffer::HostValues, Buffer::HostEdits, Buffer::Variables,
      Buffer::DirtyEvents, Buffer::EventWeights, Buffer::Control, Buffer::BinFactors});
}

void Backends::MpsEventPropagation::ensureEventWeights() {
  if( _eventWeightsValid_ ){ return; }
  auto& control = *static_cast<DevicePropagationControl*>(buffer(Buffer::Control).contents);
  const auto previous = control;
  control.factorEventWeights = control.updateVariables = control.forceBinning = control.firstPropagation = control.editCount = 0;
  control.updateWeights = 1;
  auto command = [_queue_ commandBuffer];
  auto encoder = [command computeCommandEncoder];
  try{
    encodeEventWeights(encoder);
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
  }
  catch(...){
    control = previous;
    throw;
  }
  control = previous;
  LogThrowIf(command.status != MTLCommandBufferStatusCompleted, "Metal event weight materialization failed.");
  _eventWeightsValid_ = true;
}

void Backends::MpsEventPropagation::propagate(const PropagationInputs& inputs_, PropagationResult& result_, BackendTimingSummary& timing_) {
  _diagnosticStageGpuSeconds_.clear();
  const auto& model = *_model_;
  inputs_.validate(model, _dialCount_, [&](std::size_t index_){
    return std::binary_search(_producerDials_.begin(), _producerDials_.end(), index_);
  });
  // Publications are borrowed. Snapshot even when values/generations did not
  // change, and reject missing rows on the next gradient rather than reusing
  // an older token's Jacobian. Value-only propagation needs no Jacobian.
  _producerGradientsValid_ = true;
  visitProducerGradientPublications(_dialCount_, inputs_, _producerDials_,
      [&](std::size_t i_, std::size_t){ return _producerInputCounts_[i_]; },
      [&](std::size_t i_, std::size_t, const std::vector<double>* row_, bool valid_){
    _producerGradientsValid_ = _producerGradientsValid_ and valid_;
    if( row_ ) _producerGradients_[i_] = *row_;
    else _producerGradients_[i_].clear();
  });
  _cache_.prepare(inputs_);
  uploadParameters<ObservableScalar>(buffer(Buffer::Parameters), inputs_.parameters, timing_);
  uploadExternalWeights<ObservableScalar>(buffer(Buffer::ExternalWeights), _externalGenerations_, inputs_, timing_);
  auto& control = *static_cast<DevicePropagationControl*>(buffer(Buffer::Control).contents);
  control.error = control.rebinnedEvents = control.candidates = control.editCount = 0;
  control.rebuildIndex = not _valid_;
  control.firstPropagation = not _valid_;
  bool updateWeights{true};
  bool updateResidualWeights{true};
  auto dialCount = prepareDialCache(inputs_, updateWeights, updateResidualWeights);
  _lastEvaluatedDials_ = dialCount;
  control.updateVariables = _cache_.variableInputsChanged();
  control.forceBinning = inputs_.forceBinningUpdate;
  control.eventCount = DeviceIndex(model.events.size());
  control.histogramBins = _packed_.histogramBins;
  control.dialCount = DeviceIndex(dialCount);
  control.sampleCount = DeviceIndex(model.samples.size());
  control.hasReweightCap = model.hasReweightCap;
  control.maxReweight = ObservableScalar(model.maxReweight);
  std::memset(buffer(Buffer::ChangedSamples).contents, 0, std::max(std::size_t(1), model.samples.size()) * sizeof(DeviceIndex));
  if( not inputs_.hostVariables.empty() and (_cache_.hostVariablesChanged() or inputs_.forceBinningUpdate) ){
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
  const bool rebin = _cache_.needsUpdate(inputs_) or not _valid_;
  const bool factor = usesBinFactorization();
  const bool updatePrediction = updateWeights or rebin;
  const bool rebuildHistograms = rebin or (factor ? updateResidualWeights : updateWeights);
  control.factorEventWeights = factor;
  control.updateWeights = updateWeights or (factor and rebin);
  _lastUpdatedWeights_ = control.updateWeights and rebuildHistograms;
  _lastReusedResidualHistograms_ = factor and updatePrediction and not rebuildHistograms;
  _lastUpdatedPrediction_ = updatePrediction;
  auto encodeStart = std::chrono::steady_clock::now();
  auto command = [_queue_ commandBuffer];
  auto encoder = [command computeCommandEncoder];
  dispatch(encoder, Kernel::Dials, dialCount, {Buffer::Dials, Buffer::Parameters, Buffer::Payloads, Buffer::ExternalWeights, Buffer::Responses, Buffer::Control, Buffer::DialIndices, Buffer::DialRanges, Buffer::DialInputs, Buffer::DialInputRanges,
      Buffer::FormulaPrograms, Buffer::FormulaInstructions, Buffer::FormulaConstants, Buffer::ProgramInputs,
      Buffer::CompositePrograms, Buffer::CompositeInstructions, Buffer::ProgramScratch});
  if( rebuildHistograms ){ encodeEventWeights(encoder); }
  if( rebin ){
    dispatch(encoder, Kernel::Binning, model.events.size(), {Buffer::Events, Buffer::Samples, Buffer::Nodes, Buffer::Bounds, Buffer::NodeBins,
      Buffer::Bins, Buffer::Intervals, Buffer::Variables, Buffer::DirtyEvents, Buffer::HardBins, Buffer::EventCounts, Buffer::Contributions, Buffer::ChangedSamples, Buffer::Control, Buffer::ReusableBins});
    dispatch(encoder, Kernel::ClearCounts, _packed_.histogramBins, {Buffer::BinCounts, Buffer::Control});
    dispatch(encoder, Kernel::CountContributions, model.events.size(), {Buffer::Events, Buffer::EventCounts, Buffer::Contributions, Buffer::BinCounts, Buffer::Control});
    dispatch(encoder, Kernel::IndexChunks, 1, {Buffer::BinCounts, Buffer::BinPositions, Buffer::ChunkOffsets, Buffer::Chunks, Buffer::Control});
    dispatch(encoder, Kernel::Scatter, model.events.size(), {Buffer::Events, Buffer::EventCounts, Buffer::Contributions, Buffer::BinPositions, Buffer::BinContributions, Buffer::Control});
  }
  if( rebuildHistograms ){
    const bool parallel = _pipelines_[std::size_t(Kernel::ReduceParallel)].threadExecutionWidth == 32;
    dispatch(encoder, parallel ? Kernel::ReduceParallel : Kernel::Reduce, std::size_t(_packed_.histogramChunkCapacity) * (parallel ? 32 : 1), {Buffer::Chunks, Buffer::BinContributions, Buffer::Contributions,
      Buffer::EventWeights, Buffer::PartialSums, Buffer::PartialSquares, Buffer::Control});
    dispatch(encoder, Kernel::Finish, _packed_.histogramBins, {Buffer::ChunkOffsets, Buffer::PartialSums, Buffer::PartialSquares,
      factor ? Buffer::ResidualSums : Buffer::HistogramSums, factor ? Buffer::ResidualSquares : Buffer::HistogramSquares, Buffer::Control});
  }
  // Residual sums stay resident across updates to the common factors.
  if( updatePrediction ){
    if( factor ){
      dispatch(encoder, Kernel::ApplyBinFactors, _packed_.histogramBins, {Buffer::BinFactors, Buffer::Responses,
          Buffer::ResidualSums, Buffer::ResidualSquares, Buffer::HistogramSums, Buffer::HistogramSquares, Buffer::Control});
    }
    if( _deviceLikelihood_ ){
      dispatch(encoder, Kernel::LikelihoodPartials, _likelihoodWork_.partialCount,
          {Buffer::LikelihoodBins, Buffer::HistogramSums, Buffer::HistogramSquares, Buffer::LikelihoodPartials, Buffer::LikelihoodCount});
      dispatch(encoder, Kernel::LikelihoodFinish, GUNDAM_LIKELIHOOD_REDUCTION_WIDTH, {Buffer::LikelihoodPartials, Buffer::LikelihoodTotal, Buffer::LikelihoodCount});
    }
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
  if( control.updateWeights ){ _eventWeightsValid_ = not factor; }
  if( _lastReusedResidualHistograms_ ){ ++_binFactorReuseCount_; }
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
  timing_.uniqueDialCount = _dialCount_;
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
  ensureEventWeights();
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

Backends::BackendDeviceView Backends::MpsEventPropagation::getDeviceView() {
  ensureEventWeights();
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
                  {"rebuiltContributionIndex", _lastRebuiltIndex_}, {"updatedPrediction", _lastUpdatedPrediction_},
                  {"reusedResidualHistograms", _lastReusedResidualHistograms_}}.dump();
}

std::string Backends::MpsEventPropagation::getDiagnostics() const {
  JsonType out;
  out["likelihoodExecution"] = _deviceLikelihood_ ? "device" : "host";
  out["likelihoodBinsPerThread"] = _likelihoodWork_.binsPerThread;
  out["likelihoodPartialCount"] = _likelihoodWork_.partialCount;
  out["likelihoodReductionWidth"] = GUNDAM_LIKELIHOOD_REDUCTION_WIDTH;
  out["updatedPrediction"] = _lastUpdatedPrediction_;
  out["evaluatedDials"] = _lastEvaluatedDials_;
  out["updatedEventWeights"] = _lastUpdatedWeights_;
  out["rebuiltContributionIndex"] = _lastRebuiltIndex_;
  out["dialCacheEnabled"] = _enableDialCache_;
  out["binFactorizationEnabled"] = usesBinFactorization();
  out["factoredBins"] = _factoredBinCount_;
  out["factoredParameters"] = _factoredParameterCount_;
  out["binFactorReuseCount"] = _binFactorReuseCount_;
  out["parallelHistogramsEnabled"] = _pipelines_[std::size_t(Kernel::ReduceParallel)].threadExecutionWidth == 32;
  out["lastStageGpuSeconds"] = _diagnosticStageGpuSeconds_;
  out["reusableBins"] = std::count_if(_packed_.reusableBins.begin(), _packed_.reusableBins.end(),
      [](DeviceIndex position_){ return position_ != DeviceIndex(-1); });
  out["contributionCapacity"] = _packed_.contributionCapacity;
  out["histogramChunkCapacity"] = _packed_.histogramChunkCapacity;
  out["buffers"] = JsonType::array();
  const char* names[]{"Events", "Samples", "Dials", "DialInputs", "DialInputRanges", "Targets", "Intervals", "Bins", "Nodes", "Bounds", "NodeBins", "ReusableBins", "WeightDialIndices", "VariableDialIndices", "Payloads", "Parameters", "ExternalWeights", "Responses", "DialIndices", "DialRanges", "Variables", "HostValues", "HostEdits", "DirtyEvents", "HardBins", "EventCounts", "Contributions", "ChangedSamples", "Control", "EventWeights", "BinCounts", "BinPositions", "BinContributions", "ChunkOffsets", "Chunks", "BinFactors", "ResidualSums", "ResidualSquares", "PartialSums", "PartialSquares", "HistogramSums", "HistogramSquares", "ReadbackSums", "ReadbackSquares", "ReadbackVariables", "ReadbackBins", "ReadbackCounts", "ReadbackContributions", "ReadbackWeights", "FormulaPrograms", "FormulaInstructions", "FormulaConstants", "ProgramInputs", "CompositePrograms", "CompositeInstructions", "ProgramScratch", "LikelihoodBins", "LikelihoodCount", "LikelihoodPartials", "LikelihoodTotal", "LikelihoodReadback", "GradientDials", "GradientEvents", "GradientBins", "GradientIndices", "GradientOffsets", "GradientOccurrences", "GradientResponses", "GradientScales", "GradientContributions", "GradientTotal", "GradientReadback", "GradientBindings", "GradientRanges", "GradientInputOffsets", "GradientScratch", "GradientExternal", "GradientWork"};
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

bool Backends::MpsEventPropagation::hasGradientBuffers() const {
  return buffer(Buffer::GradientWork) != nil;
}

void Backends::MpsEventPropagation::prepareGradient(const EngineView& view_) {
  if( hasGradientBuffers() ){ return; }
  MpsGradientModel model;
  model.build(view_);
  _gradientWork_ = model.work;
  auto upload = [&](Buffer slot_, const auto& data_){
    LogThrowIf(data_.size() > _device_.maxBufferLength / sizeof(typename std::decay_t<decltype(data_)>::value_type),
               "MPS gradient upload exceeds Metal maxBufferLength.");
    _buffers_[std::size_t(slot_)] = data_.empty() ? makePrivateEmptyBuffer(_device_, 8)
        : makePrivateBuffer(_device_, _queue_, data_);
    LogThrowIf(buffer(slot_) == nil, "MPS gradient upload failed.");
  };
  auto allocate = [&](Buffer slot_, std::size_t bytes_, bool shared_ = false){
    bytes_ = std::max(std::size_t(8), bytes_);
    LogThrowIf(bytes_ > _device_.maxBufferLength, "MPS gradient allocation exceeds Metal maxBufferLength.");
    _buffers_[std::size_t(slot_)] = shared_ ? makeSharedEmptyBuffer(_device_, bytes_) : makePrivateEmptyBuffer(_device_, bytes_);
    LogThrowIf(buffer(slot_) == nil, "MPS gradient allocation failed.");
  };
  try{
    upload(Buffer::GradientDials, model.inputs); upload(Buffer::GradientEvents, model.events);
    upload(Buffer::GradientBins, model.bins);
    upload(Buffer::GradientIndices, model.indices); upload(Buffer::GradientOffsets, model.parameterOffsets);
    upload(Buffer::GradientOccurrences, model.parameterOccurrences);
    upload(Buffer::GradientBindings, model.bindings); upload(Buffer::GradientRanges, model.ranges);
    upload(Buffer::GradientInputOffsets, model.inputOffsets);
    allocate(Buffer::GradientScratch, model.scratchSize*2*sizeof(ObservableScalar));
    allocate(Buffer::GradientExternal, model.externalCount*sizeof(ObservableScalar), true);
    allocate(Buffer::GradientResponses, model.inputs.size()*sizeof(ObservableScalar));
    allocate(Buffer::GradientScales, model.bins.size()*2*sizeof(ObservableScalar));
    allocate(Buffer::GradientContributions, model.contributionCount*sizeof(ObservableScalar));
    allocate(Buffer::GradientTotal, model.work.parameters*sizeof(ObservableScalar));
    allocate(Buffer::GradientReadback, model.work.parameters*sizeof(ObservableScalar), true);
    // Publish the completed layout only after all other allocations succeeded.
    upload(Buffer::GradientWork, std::vector<GradientWork<ObservableScalar>>{model.work});
  }
  catch(...){
    for( std::size_t i = std::size_t(Buffer::GradientDials); i < std::size_t(Buffer::Count); ++i ) releaseBuffer(_buffers_[i]);
    throw;
  }
}

std::vector<double> Backends::MpsEventPropagation::getLikelihoodGradient(const EngineView& view_) {
  LogThrowIf(not _valid_, "MPS gradient has no valid propagation.");
  LogThrowIf(not _producerGradientsValid_, "MPS gradient requires current, finite external/table Jacobians with matching input dimensions.");
  prepareGradient(view_);
  ensureEventWeights();
  auto* external = static_cast<ObservableScalar*>(buffer(Buffer::GradientExternal).contents);
  std::size_t externalPosition = 0;
  for( const auto& row : _producerGradients_ ) for( auto slope : row ) external[externalPosition++] = ObservableScalar(slope);
  auto command = [_queue_ commandBuffer];
  auto encoder = [command computeCommandEncoder];
  dispatch(encoder, Kernel::GradientDials, _gradientWork_.dials,
      {Buffer::GradientDials, Buffer::Dials, Buffer::GradientBindings, Buffer::GradientRanges,
       Buffer::Parameters, Buffer::Payloads, Buffer::ExternalWeights, Buffer::GradientExternal,
       Buffer::FormulaPrograms, Buffer::FormulaInstructions, Buffer::FormulaConstants, Buffer::ProgramInputs,
       Buffer::CompositePrograms, Buffer::CompositeInstructions, Buffer::GradientScratch, Buffer::GradientResponses, Buffer::GradientWork});
  dispatch(encoder, Kernel::GradientBins, _gradientWork_.bins,
      {Buffer::GradientBins, Buffer::HistogramSums, Buffer::HistogramSquares, Buffer::GradientScales, Buffer::GradientWork});
  dispatch(encoder, Kernel::GradientEvents, _gradientWork_.events,
      {Buffer::GradientEvents, Buffer::GradientIndices, Buffer::GradientInputOffsets, Buffer::Responses,
       Buffer::GradientResponses, Buffer::GradientScales, Buffer::EventWeights, Buffer::GradientContributions, Buffer::GradientWork,
       Buffer::Events, Buffer::Samples, Buffer::Targets, Buffer::VariableDialIndices, Buffer::Variables,
       Buffer::EventCounts, Buffer::Contributions, Buffer::Nodes, Buffer::Bounds, Buffer::NodeBins, Buffer::Bins, Buffer::Intervals});
  dispatch(encoder, Kernel::GradientReduce, std::size_t(_gradientWork_.parameters)*128,
      {Buffer::GradientOffsets, Buffer::GradientOccurrences, Buffer::GradientContributions, Buffer::GradientTotal});
  [encoder endEncoding];
  auto blit = [command blitCommandEncoder];
  if( _gradientWork_.parameters ) copy(blit, Buffer::GradientTotal, Buffer::GradientReadback, _gradientWork_.parameters*sizeof(ObservableScalar));
  [blit endEncoding];
  [command commit]; [command waitUntilCompleted];
  LogThrowIf(command.status != MTLCommandBufferStatusCompleted, "MPS gradient command failed.");
  const auto* values = static_cast<const ObservableScalar*>(buffer(Buffer::GradientReadback).contents);
  std::vector<double> gradient(_gradientWork_.parameters);
  for( std::size_t i = 0; i < gradient.size(); ++i ){
    gradient[i] = double(values[i].high) + values[i].low;
    LogThrowIf(not std::isfinite(gradient[i]), "Non-finite MPS likelihood gradient.");
  }
  return gradient;
}
