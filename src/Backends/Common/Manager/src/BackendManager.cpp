#include "BackendManager.h"

#include "LikelihoodInterface.h"
#include "JointProbabilityBase.h"
#include "ExternalWeightBuffer.h"
#include "DialInputBuffer.h"
#include "Logger.h"
#include "Parameter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>

std::string Backends::formatBackendTimingSummary(const BackendTimingSummary& timing_) {
  std::stringstream ss;
  ss << "Backend timing summary:"
     << " build[scan=" << timing_.buildCompatibilityScanSeconds
     << "s, parLookup=" << timing_.buildParameterLookupSeconds
     << "s, firstPass=" << timing_.buildFirstPassSeconds
     << "s, secondPass=" << timing_.buildSecondPassSeconds
     << "s, finalFlatten=" << timing_.buildFinalFlattenSeconds
     << "s, histIndex=" << timing_.buildHistogramIndexSeconds
     << "s, bufferUpload=" << timing_.buildBufferUploadSeconds
     << "s]"
     << " device[paramUpload=" << timing_.parameterUploadSeconds
     << "s, externalWeightUpload=" << timing_.externalWeightUploadSeconds
     << "s/" << timing_.externalWeightUploadBytes << "B/" << timing_.externalWeightUploadBlocks << " blocks"
     << ", cachedDialStage=" << timing_.cachedDialStageSeconds
     << "s, eventWeightsStage=" << timing_.eventWeightsStageSeconds
     << "s, variables=" << timing_.variableStageSeconds
     << "s, binning=" << timing_.binningStageSeconds
     << "s, rebinnedEvents=" << timing_.rebinnedEventCount
     << ", candidates=" << timing_.binCandidateCount
     << ", contributions=" << timing_.binContributionCount
     << ", histogramStage=" << timing_.histogramStageSeconds
     << "s, encode=" << timing_.commandEncodeSeconds
     << "s, wait=" << timing_.deviceWaitSeconds
     << "s, devicePropagation=" << timing_.devicePropagationSeconds
     << "s, stateReadback=" << timing_.eventStateReadbackSeconds << "s/" << timing_.eventStateReadbackBytes
     << "B, contributionBuffer=" << timing_.contributionBufferBytes
     << "B, histReadback=" << timing_.histogramReadbackSeconds
     << "s/" << timing_.histogramReadbackBytes
     << "B, eventWeightReadback=" << timing_.eventWeightReadbackSeconds
     << "s/" << timing_.eventWeightReadbackBytes
     << "B]"
     << " host[llh=" << timing_.likelihoodHostSeconds
     << "s, eventWeightMaterialize=" << timing_.eventWeightMaterializationSeconds
     << "s, histogramMaterialize=" << timing_.histogramMaterializationSeconds
     << "s]"
     << " counts[uniqueDials=" << timing_.uniqueDialCount
     << ", cachedDials=" << timing_.cachedDialCount
     << ", eventDialIndices=" << timing_.eventDialIndexCount
     << ", splineScalars=" << timing_.splineScalarCount
     << "]";
  return ss.str();
}

void Backends::BackendManager::configureImpl() {
  _config_.clearFields();
  _config_.defineFields({{"isEnabled"}, {"useBackend", {"type"}}, {"options"}});
  _config_.checkConfiguration();
  const auto& config = _config_.getConfig();
  LogThrowIf(not config.is_null() and not config.is_object(), "BackendManager configuration must be an object.");
  for( auto entry = config.begin(); entry != config.end(); ++entry ){
    const auto& key = entry.key();
    LogThrowIf(key != "isEnabled" and key != "useBackend" and key != "type" and key != "options" and key != "name",
               "Unsupported BackendManager field: " << key << ". Put backend settings in the named options list.");
  }
  _isEnabled_ = false;
  _type_ = "CPU";
  _backendOptions_ = JsonType::array();
  _config_.fillValue(_isEnabled_, "isEnabled");
  _config_.fillValue(_type_, "useBackend");
  _config_.fillValue(_backendOptions_, "options");
  LogThrowIf(not _backendOptions_.is_array(), "Backend options must be a list of named configurations.");
}

bool Backends::BackendManager::shouldMaterialize(OutputRequest outputRequest_) const {
  return std::find(_materializeOutputList_.begin(), _materializeOutputList_.end(), outputRequest_) != _materializeOutputList_.end();
}

bool Backends::BackendManager::willAutoMaterialize(OutputRequest outputRequest_) const {
  return _enableAutoMaterialize_ and shouldMaterialize(outputRequest_);
}

void Backends::BackendManager::setMaterializeOutputList(std::initializer_list<OutputRequest> materializeOutputList_) {
  setMaterializeOutputList(std::vector<OutputRequest>(materializeOutputList_));
}

void Backends::BackendManager::materialize(OutputRequest outputRequest_) {
  LogThrowIf(not hasBackend(), "No backend initialized.");
  LogThrowIf(not _lastPropagationToken_.isValid, "No backend propagation token available for materialization.");
  LogThrowIf(_likelihoodInterfacePtr_ == nullptr, "BackendsManager requires a LikelihoodInterface for materialization.");

  auto* backend = getBackend();
  auto status = backend->getStatus(_lastPropagationToken_);
  auto outputState = status.state(outputRequest_);
  LogThrowIf(outputState != OutputState::ReadyOnDevice and outputState != OutputState::ReadyOnHost,
             "Requested backend output is not ready for materialization: " << outputRequest_.toString());

  if( outputState == OutputState::ReadyOnDevice ){
    backend->materialize(_lastPropagationToken_, outputRequest_);
    status = backend->getStatus(_lastPropagationToken_);
    outputState = status.state(outputRequest_);
    LogThrowIf(outputState != OutputState::ReadyOnHost and outputState != OutputState::ReadyOnDevice,
               "Backend materialization did not produce a host-ready output: " << outputRequest_.toString());
  }

  switch( outputRequest_.value ){
    case OutputRequest::EventState: {
      const auto& state = backend->getEventStateHostView(_lastPropagationToken_);
      if( _lastMaterializedStateRevision_ == state.revision ){ return; }
      const auto& model = _backendEngineLayout_->view.propagation;
      const auto& bindings = _backendEngineLayout_->bindings;
      LogThrowIf(state.variables.size() != model.variableValues.size() or state.bins.size() != model.events.size()
                 or state.contributions.size() != model.events.size(), "Backend event state size mismatch.");
      for( const auto& binding : bindings.variables ){
        if( binding.isDialTarget and _lastMaterializedVariableRevision_ != state.variableRevision ){
          binding.event->getVariables().getVarList()[binding.variableIndex].set(state.variables.at(binding.valueIndex));
        }
      }
      std::vector<bool> changedSamples(model.samples.size(), false);
      for( std::size_t i = 0; i < model.samples.size(); ++i ){
        changedSamples[i] = i >= _lastMaterializedSampleRevisions_.size()
            or _lastMaterializedSampleRevisions_[i] != state.sampleRevisions.at(i);
      }
      // Host pointer lists are diagnostic state. Rebuild only on explicit/auto
      // materialization, and only when the cached event state changed.
      for( std::size_t i = 0; i < model.samples.size(); ++i ){
        const auto& sample = model.samples[i];
        if( not changedSamples[i] or (not sample.useDynamicBinning and not sample.binning.isSmoothed) ){ continue; }
        auto* histogram = bindings.samples.at(i).histogram;
        for( auto& bin : histogram->getBinContextList() ){
          bin.eventPtrList.clear();
          bin.eventContributionList.clear();
        }
        histogram->getOutsideContributionList().clear();
      }
      for( std::size_t i = 0; i < model.events.size(); ++i ){
        const auto& event = model.events[i];
        auto* eventPtr = bindings.events.at(i).event;
        const auto& sample = model.samples.at(event.samplePosition);
        if( not changedSamples[event.samplePosition] or (not sample.useDynamicBinning and not sample.binning.isSmoothed) ){ continue; }
        int bin = state.bins.at(event.resultIndex);
        eventPtr->getIndices().bin = bin;
        auto* histogram = bindings.samples.at(event.samplePosition).histogram;
        if( bin >= 0 ){ histogram->getBinContextList().at(bin).eventPtrList.emplace_back(eventPtr); }
        if( sample.binning.isSmoothed ){
          for( const auto& contribution : state.contributions[event.resultIndex] ){
            auto& list = contribution.binIndex < 0 ? histogram->getOutsideContributionList()
                : histogram->getBinContextList().at(contribution.binIndex).eventContributionList;
            list.push_back({eventPtr, contribution.fraction});
          }
        }
      }
      _likelihoodInterfacePtr_->getModelPropagator().invalidateVariableResponses();
      _lastMaterializedStateRevision_ = state.revision;
      _lastMaterializedVariableRevision_ = state.variableRevision;
      _lastMaterializedSampleRevisions_ = state.sampleRevisions;
      return;
    }
    case OutputRequest::EventWeights: {
      const auto& eventWeights = backend->getEventWeightsHostView(_lastPropagationToken_);
      const auto& model = _backendEngineLayout_->view.propagation;
      const auto& eventBindings = _backendEngineLayout_->bindings.events;
      LogThrowIf(eventWeights.size() != model.events.size(), "Event weights host view size mismatch.");
      LogThrowIf(eventBindings.size() != model.events.size(), "Event bindings size mismatch.");
      for( std::size_t iEvent = 0 ; iEvent < model.events.size() ; iEvent++ ){
        const auto& event = model.events.at(iEvent);
        auto* eventPtr = eventBindings.at(iEvent).event;
        LogThrowIf(eventPtr == nullptr, "Null event binding during backend materialization.");
        eventPtr->getWeights().current = eventWeights.at(event.resultIndex);
      }
      return;
    }

    case OutputRequest::Histograms: {
      const auto& model = _backendEngineLayout_->view.propagation;
      const auto& sampleBindings = _backendEngineLayout_->bindings.samples;
      const auto& histSums = backend->getHistogramSumsHostView(_lastPropagationToken_);
      const auto& histSumSquares = backend->getHistogramSumSquaresHostView(_lastPropagationToken_);
      const auto& outsideSums = backend->getOutsideSumsHostView(_lastPropagationToken_);
      const auto& outsideSumSquares = backend->getOutsideSumSquaresHostView(_lastPropagationToken_);
      LogThrowIf(histSums.size() != std::size_t(model.totalBins), "Histogram sums host view size mismatch.");
      LogThrowIf(histSumSquares.size() != std::size_t(model.totalBins), "Histogram sum squares host view size mismatch.");
      LogThrowIf(sampleBindings.size() != model.samples.size(), "Sample bindings size mismatch.");

      for( std::size_t iSample = 0 ; iSample < model.samples.size() ; iSample++ ){
        const auto& sample = model.samples.at(iSample);
        auto* histogramPtr = sampleBindings.at(iSample).histogram;
        LogThrowIf(histogramPtr == nullptr, "Null histogram binding during backend materialization.");
        if( sample.binning.isSmoothed ){
          auto& outside = histogramPtr->getOutsideBinContent();
          outside.sumWeights = outsideSums.at(iSample);
          outside.sqrtSumSqWeights = std::sqrt(outsideSumSquares.at(iSample));
        }
        auto& binContentList = histogramPtr->getBinContentList();
        auto& binContextList = histogramPtr->getBinContextList();

        for( auto& binContent : binContentList ){
          binContent.sumWeights = 0;
          binContent.sqrtSumSqWeights = 0;
        }

        for( auto& binContext : binContextList ){
          int globalBin = sample.binOffset + binContext.bin.getIndex();
          auto& binContent = binContentList[binContext.bin.getIndex()];
          binContent.sumWeights = histSums.at(globalBin);
          binContent.sqrtSumSqWeights = std::sqrt(histSumSquares.at(globalBin));
        }
      }
      return;
    }

    case OutputRequest::SampleLikelihoods:
      LogThrow("BackendsManager::materialize(OutputRequest::SampleLikelihoods) is not implemented yet: LikelihoodInterface has no destination slot yet.");
      return;

    case OutputRequest::StatLikelihood:
      _likelihoodInterfacePtr_->getBuffer().statLikelihood = backend->getLikelihood(_lastPropagationToken_);
      return;
    default:
      LogError << "Unhandled OutputRequest in BackendsManager::materialize: " << outputRequest_.toString() << std::endl;
  }

  LogThrow("Unhandled OutputRequest in BackendsManager::materialize: " << outputRequest_.toString());
}

void Backends::BackendManager::initializeImpl() {
  deactivateBackend();
  _availableBackendList_.clear();
  const auto names = getCompiledBackendNames();
  std::vector<std::string> configuredNames;
  for( const auto& option : _backendOptions_ ){
    LogThrowIf(not option.is_object() or not option.contains("name") or not option.at("name").is_string(),
               "Each backend option must be an object with a string name.");
    const auto name = option.at("name").get<std::string>();
    LogThrowIf(std::find(configuredNames.begin(), configuredNames.end(), name) != configuredNames.end(),
               "Duplicate backend options: " << name);
    configuredNames.push_back(name);
    LogThrowIf(name != "CPU" and name != "CUDA" and name != "MPS", "Unknown backend: " << name);
  }
  for( const auto& name : names ){
    JsonType options = JsonType::object();
    for( const auto& option : _backendOptions_ ){
      if( option.at("name") == name ){ options = option; options.erase("name"); }
    }
    auto backend = makeBackend(name);
    ConfigReader config(options);
    backend->configure(config);
    _availableBackendList_.push_back({name, config, std::move(backend)});
  }
  // Capture the prior mask before any parameter scans or backend switches. A
  // bin emptied by a later prediction must never become an ignored prior bin.
  _ignoredBinsAtPrior_.clear();
  if( _likelihoodInterfacePtr_ != nullptr ){
    for( const auto& sample : _likelihoodInterfacePtr_->getSamplePairList() ){
      std::vector<bool> ignored;
      for( const auto& bin : sample.model->getHistogram().getBinContentList() ){
        ignored.push_back(_likelihoodInterfacePtr_->getJointProbabilityPtr()->isIgnoreBinsWithZeroPredictionAtPrior()
                          and bin.sumWeights == 0.);
      }
      _ignoredBinsAtPrior_.push_back(std::move(ignored));
    }
  }
  if( _isEnabled_ ){ activate(_type_); }
}

void Backends::BackendManager::updateDataHistograms() {
  _isGradientValidated_ = false;
  if( not hasBackend() ){ return; }
  LogThrowIf(_likelihoodInterfacePtr_ == nullptr, "No likelihood interface for data histogram updates.");
  auto& samples = _backendEngineLayout_->view.likelihood.samples;
  const auto& pairs = _likelihoodInterfacePtr_->getSamplePairList();
  LogThrowIf(samples.size() != pairs.size(), "Cannot change the data sample list after backend initialization.");
  std::vector<std::vector<double>> contents(samples.size());
  // Validate all dimensions and allocate before touching the current snapshot.
  for( std::size_t i = 0; i < samples.size(); ++i ){
    const auto& bins = pairs[i].data->getHistogram().getBinContentList();
    LogThrowIf(bins.size() != samples[i].dataSums.size(), "Cannot change the data bin count after backend initialization.");
    contents[i].reserve(bins.size());
    for( const auto& bin : bins ){ contents[i].push_back(bin.sumWeights); }
  }
  // Wait without materializing: host histograms may already contain a new toy.
  if( _lastPropagationToken_.isValid ){ _activeBackendPtr_->wait(_lastPropagationToken_); }
  ++_backendGeneration_; // Invalidate deferred results made with the previous data.
  _lastPropagationToken_ = {};
  // A native toy generator may also have moved host event state. The next
  // explicit materialization must not skip restoring an unchanged device state.
  _lastMaterializedStateRevision_ = _lastMaterializedVariableRevision_ = 0;
  _lastMaterializedSampleRevisions_.clear();
  for( std::size_t i = 0; i < samples.size(); ++i ){ samples[i].dataSums.swap(contents[i]); }
  try{
    _activeBackendPtr_->updateDataHistograms(
        std::shared_ptr<const EngineView>(_backendEngineLayout_, &_backendEngineLayout_->view));
  }
  catch(...){
    // Fail closed if an upload fails; the next use must explicitly reactivate.
    // In particular, never silently fit the previous toy after a failed update.
    _activeBackendPtr_ = nullptr;
    _activeBackendName_.clear();
    throw;
  }
}

void Backends::BackendManager::buildEngineLayout() {
  _isGradientValidated_ = false;
  LogThrowIf(_likelihoodInterfacePtr_ == nullptr, "BackendManager requires a LikelihoodInterface before activation.");
  LogInfo << "Preparing backend model on CPU (events, dials, binning and likelihood)..." << std::endl;
  const auto start = std::chrono::steady_clock::now();
  auto layout = std::make_shared<EngineLayout>();
  layout->build(*_likelihoodInterfacePtr_);
  LogThrowIf(layout->view.likelihood.samples.size() != _ignoredBinsAtPrior_.size(),
             "Cannot change the likelihood sample list after initialization.");
  for( std::size_t i = 0; i < _ignoredBinsAtPrior_.size(); ++i ){
    auto& sample = layout->view.likelihood.samples[i];
    LogThrowIf(sample.ignoredBins.size() != _ignoredBinsAtPrior_[i].size(),
               "Cannot change the likelihood bin count after initialization.");
    sample.ignoredBins = _ignoredBinsAtPrior_[i];
  }

  _binningUpdateListeners_.clear();
  _hostBinningChanged_ = std::make_shared<bool>(false);
  for( const auto& binding : layout->bindings.samples ){
    _binningUpdateListeners_.push_back(binding.sample->addBinningUpdateListener(
        [changed = _hostBinningChanged_]{ *changed = true; }));
  }
  const auto& model = layout->view.propagation;
  LogInfo << "Backend model ready: " << model.events.size() << " events, "
          << model.dials.size() << " dials, " << model.parameterCount << " parameters, "
          << model.samples.size() << " samples, " << model.totalBins << " bins ("
          << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s)." << std::endl;
  _backendEngineLayout_ = std::move(layout);
}

std::string Backends::BackendManager::getActiveBackendName() const { return _activeBackendName_; }

std::vector<std::string> Backends::BackendManager::getAvailableBackendNames() const {
  std::vector<std::string> names;
  for( const auto& entry : _availableBackendList_ ){ names.push_back(entry.name); }
  return names;
}

void Backends::BackendManager::deactivateBackend() {
  _isGradientValidated_ = false;
  if( hasBackend() ){
    if( _lastPropagationToken_.isValid ){
      _activeBackendPtr_->wait(_lastPropagationToken_);
      const auto status = _activeBackendPtr_->getStatus(_lastPropagationToken_);
      for( auto output : {OutputRequest::EventState, OutputRequest::EventWeights,
                         OutputRequest::Histograms, OutputRequest::StatLikelihood} ){
        const auto state = status.state(output);
        if( state == OutputState::ReadyOnDevice or state == OutputState::ReadyOnHost ){ materialize(output); }
      }
    }
    _activeBackendPtr_ = nullptr;
    for( auto& entry : _availableBackendList_ ){
      if( entry.name != _activeBackendName_ ){ continue; }
      entry.backend.reset(); // Release all device buffers before constructing another backend.
      entry.backend = makeBackend(entry.name);
      entry.backend->configure(entry.config);
    }
    _activeBackendName_.clear();
    if( _likelihoodInterfacePtr_ != nullptr ){
      auto& propagator = _likelihoodInterfacePtr_->getModelPropagator();
      propagator.invalidateVariableResponses();
      propagator.getDialManager().invalidateInputBuffers();
    }
  }
  ++_backendGeneration_;
  _lastPropagationToken_ = {};
  _lastPreparationSeconds_ = _lastSnapshotSeconds_ = _lastHostVariableReadSeconds_ = 0.;
  _lastHostVariableReadCount_ = 0;
  _lastMaterializedStateRevision_ = _lastMaterializedVariableRevision_ = 0;
  _lastMaterializedSampleRevisions_.clear();
  _binningUpdateListeners_.clear();
  _hostBinningChanged_.reset();
  _backendEngineLayout_ = std::make_shared<EngineLayout>(); // Release the old flattened host model as well.
}

void Backends::BackendManager::activate(const std::string& name_) { activateImpl(name_, nullptr); }

void Backends::BackendManager::activate(const std::string& name_, const ConfigReader& config_) {
  activateImpl(name_, &config_);
}

void Backends::BackendManager::activateImpl(const std::string& name_, const ConfigReader* config_) {
  auto found = std::find_if(_availableBackendList_.begin(), _availableBackendList_.end(),
                            [&](const auto& entry_){ return entry_.name == name_; });
  LogThrowIf(found == _availableBackendList_.end(), "Backend is not available in this build: " << name_);
  // Validate overrides before releasing the active backend. Configuration never allocates device buffers.
  const auto config = config_ == nullptr ? found->config : *config_;
  LogInfo << "Initializing " << name_ << " backend..." << std::endl;
  const auto start = std::chrono::steady_clock::now();
  auto replacement = makeBackend(name_);
  replacement->configure(config);
  deactivateBackend();
  buildEngineLayout(); // Rebuild from in-memory host state, including observable and binning changes.
  // Publish only after a successful build; failures release partially allocated buffers.
  replacement->buildShared(std::shared_ptr<const EngineView>(_backendEngineLayout_, &_backendEngineLayout_->view));
  found->config = config;
  found->backend = std::move(replacement);
  _activeBackendPtr_ = found->backend.get();
  _activeBackendName_ = name_;
  _type_ = name_;
  LogInfo << "Backend " << _activeBackendPtr_->getName() << " ready ("
          << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s)." << std::endl;
}

void Backends::BackendManager::replaceBackend(const std::string& type_, const ConfigReader& backendConfig_) {
  activate(type_, backendConfig_);
}

std::future<Backends::BackendPropagationResult> Backends::BackendManager::propagate(bool withGradients_) {
  LogThrowIf(not hasBackend(), "No backend initialized.");

  LogThrowIf(_likelihoodInterfacePtr_ == nullptr, "No likelihood interface for backend propagation.");
  auto prepareStart = std::chrono::steady_clock::now();
  auto& propagator = _likelihoodInterfacePtr_->getModelPropagator();
  const bool handlesEventState = _activeBackendPtr_->getCapabilities().supportsObservableTransforms;
  propagator.preparePropagation(not handlesEventState);
  _lastPreparationSeconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - prepareStart).count();

  const auto snapshotStart = std::chrono::steady_clock::now();
  _lastSnapshotSeconds_ = _lastHostVariableReadSeconds_ = 0.;
  _lastHostVariableReadCount_ = 0;
  Backends::PropagationInputs inputs;
  auto& snapshot = inputs.parameters;
  snapshot.values.reserve(_backendEngineLayout_->bindings.parameters.size());
  for( const auto& binding : _backendEngineLayout_->bindings.parameters ){
    LogThrowIf(binding.parameter == nullptr, "Null parameter binding while building backend snapshot.");
    snapshot.values.emplace_back(binding.parameter->getParameterValue());
  }

  inputs.externalWeights.reserve(_backendEngineLayout_->bindings.externalWeights.size());
  for( std::size_t iBlock = 0; iBlock < _backendEngineLayout_->bindings.externalWeights.size(); ++iBlock ){
    auto& binding = _backendEngineLayout_->bindings.externalWeights[iBlock];
    const auto& source = binding.source;
    const auto& layout = _backendEngineLayout_->view.propagation.externalWeightBlocks[iBlock];
    if( binding.table != nullptr ){
      inputs.externalWeights.push_back({binding.table->data(), binding.table->size(), layout.offset, ++binding.generation});
    }
    else{
      inputs.externalWeights.push_back({source->data(), source->size(), layout.offset, source->getGeneration()});
    }
  }
  const auto& model = _backendEngineLayout_->view.propagation;
  if( withGradients_ ){
    const auto& bindings = _backendEngineLayout_->bindings.gradientDials;
    inputs.externalDialGradientIndices.reserve(bindings.size());
    inputs.externalDialGradients.reserve(bindings.size());
    for( const auto& binding : bindings ){
      LogThrowIf(binding.inputs == nullptr, "Missing external gradient input buffer.");
      inputs.externalDialGradientIndices.push_back(binding.dialIndex);
      inputs.externalDialGradients.push_back(binding.dial->evalGradients(*binding.inputs));
    }
  }
  inputs.forceBinningUpdate = *_hostBinningChanged_;
  inputs.variableInputRevisions.reserve(_backendEngineLayout_->bindings.variableInputBuffers.size());
  for( const auto* buffer : _backendEngineLayout_->bindings.variableInputBuffers ){
    inputs.variableInputRevisions.emplace_back(buffer->getRevision());
  }
  const auto& variableBindings = _backendEngineLayout_->bindings.variables;
  // The layout owns the loaded observable baseline; executors keep their current
  // state resident. Only an explicit host binning update publishes edited values.
  if( handlesEventState and inputs.forceBinningUpdate ){
    const auto hostReadStart = std::chrono::steady_clock::now();
    inputs.hostVariables = model.variableValues;
    for( const auto& binding : variableBindings ){
      inputs.hostVariables.at(binding.valueIndex) = binding.event->getVariables().getVarList()[binding.variableIndex].getVarAsDouble();
      ++_lastHostVariableReadCount_;
    }
    _lastHostVariableReadSeconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now()-hostReadStart).count();
  }
  _lastSnapshotSeconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now()-snapshotStart).count();
  auto token = _activeBackendPtr_->requestPropagation(inputs);
  if( token.isValid ){
    *_hostBinningChanged_ = false;
    if( handlesEventState ){ propagator.invalidateVariableResponses(); }
  }
  _lastPropagationToken_ = token;
  if( not token.isValid ){
    return std::async(std::launch::deferred, []{
      return BackendPropagationResult{};
    });
  }

  return std::async(std::launch::deferred, [this, token, generation = _backendGeneration_]{
    LogThrowIf(generation != _backendGeneration_, "Propagation result belongs to a replaced backend.");
    BackendPropagationResult result;
    auto* backend = getBackend();
    backend->wait(token);

    auto status = backend->getStatus(token);
    for( auto outputRequest : {OutputRequest::EventState, OutputRequest::EventWeights, OutputRequest::Histograms, OutputRequest::SampleLikelihoods, OutputRequest::StatLikelihood} ){
      auto outputState = status.state(outputRequest);
      if( outputState == OutputState::Failed ){ continue; }
      if( outputState != OutputState::ReadyOnDevice and outputState != OutputState::ReadyOnHost ){
        continue;
      }

      result.isValid = true;
      if( outputRequest == OutputRequest::StatLikelihood ){
        result.statLikelihood = backend->getLikelihood(token);
        result.hasStatLikelihood = true;
      }

      if( not willAutoMaterialize(outputRequest) ){ continue; }
      materialize(outputRequest);
    }

    return result;
  });
}

void Backends::BackendManager::validateLikelihoodGradient() const {
  LogThrowIf(not hasBackend(), "No active backend for likelihood gradients.");
  LogThrowIf(not _activeBackendPtr_->getCapabilities().supportsLikelihoodGradient,
             _activeBackendName_ << " does not support analytic likelihood gradients.");
  _activeBackendPtr_->validateLikelihoodGradient();
  // The layout owns an immutable model snapshot. Structural validation is
  // repeated on activation/rebuild, data updates or interface replacement.
  LogThrowIf(_likelihoodInterfacePtr_ == nullptr, "No likelihood interface for backend gradients.");
  if( not _isGradientValidated_ ){ _likelihoodInterfacePtr_->validateWeightGradient(false); }
  // Producer checks also request publication of the next Jacobian. They must
  // run for every gradient propagation, even when the structure is validated.
  // Formula/composite derivatives are validated by the numerical backend.
  for( const auto& binding : _backendEngineLayout_->bindings.gradientDials ){
    if( binding.dial == nullptr ){ continue; }
    LogThrowIf(not binding.dial->hasAnalyticGradient(),
               "Backend gradient is unsupported for dial " << binding.dial->getDialTypeName());
  }
  _isGradientValidated_ = true;
}

void Backends::BackendManager::evalGradient(ParameterGradient& gradient_) {
  validateLikelihoodGradient();
  LogThrowIf(_likelihoodInterfacePtr_ == nullptr, "No likelihood interface for backend gradients.");
  auto& likelihood = *_likelihoodInterfacePtr_;

  // Refresh the backend snapshot/token even when no value callback preceded us.
  auto propagation = propagate(true);
  LogThrowIf(not propagation.valid(), "Backend gradient propagation is unavailable.");
  const auto result = propagation.get();
  LogThrowIf(not result.isValid, "Backend gradient propagation failed.");
  auto statGradient = likelihood.prepareGradient(gradient_);
  likelihood.evalPenaltyLikelihood();
  if( result.hasStatLikelihood ){
    likelihood.getBuffer().statLikelihood = result.statLikelihood;
  }
  else{
    std::future<bool> invalidPropagation;
    likelihood.evalStatLikelihood(invalidPropagation);
  }
  likelihood.getBuffer().updateTotal();
  LogThrowIf(not likelihood.getBuffer().isValid(), "Cannot differentiate a non-finite likelihood.");

  evalStatLikelihoodGradient(statGradient);
  likelihood.finalizeGradient(statGradient, gradient_);
}

void Backends::BackendManager::evalStatLikelihoodGradient(ParameterGradient& gradient_) {
  const auto gradient = _activeBackendPtr_->getLikelihoodGradient(_lastPropagationToken_);
  const auto& bindings = _backendEngineLayout_->bindings.parameters;
  LogThrowIf(gradient.size() != bindings.size(), "Backend gradient dimension mismatch.");
  for( std::size_t i = 0; i < bindings.size(); ++i ){
    auto* parameter = bindings[i].parameter;
    if( parameter != nullptr and gradient_.values.count(parameter) ){
      gradient_.addValue(parameter, gradient[i]);
    }
  }
}
