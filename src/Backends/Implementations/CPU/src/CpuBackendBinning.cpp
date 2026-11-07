#include "CpuBackend.h"

#include "Semantics/BackendDialSemantics.h"
#include "Semantics/BackendVariableResponse.h"

#include "Logger.h"

#include <algorithm>
#include <cmath>
#include <limits>

void Backends::CpuBackend::initializeBinning() {
  const auto& samples = _engineView_->propagation.samples;
  _binningIndices_.resize(samples.size());
  _reusableBinPositions_.assign(samples.size(), {});
  for( std::size_t iSample = 0; iSample < samples.size(); ++iSample ){
    const auto& sample = samples[iSample];
    const auto& layout = sample.binning;
    auto& index = _binningIndices_[iSample];
    index.build(layout);
    if( not sample.useDynamicBinning or layout.isSmoothed ){ continue; }
    _reusableBinPositions_[iSample] = index.buildReusableBinPositions(layout, sample.binCount);
  }
}

double Backends::CpuBackend::evalResponse(std::size_t dial_) const {
  return _isCachedDial_[dial_] ? _cachedDialResponses_[dial_]
      : Semantics::evalDialResponse(_engineView_->propagation, _engineView_->propagation.dials[dial_], *_activeInputs_);
}

void Backends::CpuBackend::updateEventState(const PropagationInputs& inputs_) {
  const auto& model = _engineView_->propagation;
  _activeInputs_ = &inputs_;
  _eventStateCache_.prepare(inputs_);
  if( not _eventStateCache_.needsUpdate(inputs_) ){
    _lastTiming_.binContributionCount = _cachedContributionCount_;
    _activeInputs_ = nullptr;
    return;
  }
  {
    auto timer = _variableStageTimer_.scopeTime();
    runEventStateJob("CpuBackend::updateEventVariables");
  }
  {
    auto timer = _binningStageTimer_.scopeTime();
    runEventStateJob("CpuBackend::updateEventBinning");
  }
  _lastTiming_.variableStageSeconds = _variableStageTimer_.eval().count();
  _lastTiming_.binningStageSeconds = _binningStageTimer_.eval().count();
  for( const auto& worker : _binningWorkers_ ){
    _lastTiming_.rebinnedEventCount += worker.rebinnedEvents;
    _lastTiming_.binCandidateCount += worker.candidatesTested;
    _lastTiming_.binContributionCount += worker.contributionCount;
  }
  _cachedContributionCount_ = _lastTiming_.binContributionCount;
  std::vector<std::uint8_t> changedSamples(model.samples.size(), false);
  for( const auto& worker : _binningWorkers_ ){
    for( std::size_t i = 0; i < changedSamples.size(); ++i ){ changedSamples[i] |= worker.changedSamples[i]; }
  }
  _eventStateCache_.commit(inputs_, changedSamples);
  _eventStateCache_.publishRevisions(_eventState_);
  _activeInputs_ = nullptr;
}

void Backends::CpuBackend::updateEventVariablesThread(int iThread_) {
  const auto& model = _engineView_->propagation;
  const auto& inputs = *_activeInputs_;
  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(iThread_, _threadPool_.getNbThreads(), model.events.size());
  for( std::size_t iEvent = bounds.beginIndex; iEvent < bounds.endIndex; ++iEvent ){
    const auto& event = model.events[iEvent];
    if( event.samplePosition >= model.samples.size() ){ continue; }
    const auto& sample = model.samples[event.samplePosition];
    auto assign = [&](std::size_t index_, double value_){
      if( _eventState_.variables[index_] != value_ ){
        _eventState_.variables[index_] = value_;
        if( index_ - event.firstVariable < sample.binning.variableNames.size() ){ _validBinning_[event.resultIndex] = false; }
      }
    };
    if( not inputs.hostVariables.empty() and (sample.readHostVariables or inputs.forceBinningUpdate) ){
      for( std::size_t i = event.firstVariable; i < event.firstVariable + event.variableCount; ++i ){
        if( _eventStateCache_.hostVariableChanged(i, inputs) ){ assign(i, inputs.hostVariables[i]); }
      }
    }
    if( not _eventStateCache_.variableInputsChanged() ){ continue; }
    for( std::size_t i = event.firstVariableResponse; i < event.firstVariableResponse + event.variableResponseCount; ++i ){
      const auto& target = model.variableResponses[i];
      double value = Semantics::evalVariableResponse(target, model.variableDialIndices.data(),
          [this](std::size_t dial_){ return evalResponse(dial_); });
      assign(target.variableIndex, value);
    }
  }
}

void Backends::CpuBackend::updateEventBinningThread(int iThread_) {
  const auto& model = _engineView_->propagation;
  auto& worker = _binningWorkers_[iThread_];
  worker.rebinnedEvents = worker.candidatesTested = worker.contributionCount = 0;
  std::size_t contributionCount{0};
  worker.changedSamples.assign(model.samples.size(), false);
  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(iThread_, _threadPool_.getNbThreads(), model.events.size());
  for( std::size_t iEvent = bounds.beginIndex; iEvent < bounds.endIndex; ++iEvent ){
    const auto& event = model.events[iEvent];
    if( event.samplePosition >= model.samples.size() ){
      _eventState_.bins[event.resultIndex] = event.binIndex;
      continue;
    }
    const auto& sample = model.samples[event.samplePosition];
    if( not sample.useDynamicBinning and not sample.binning.isSmoothed ){
      _eventState_.bins[event.resultIndex] = event.binIndex;
      continue;
    }
    if( _validBinning_[event.resultIndex] and not sample.readHostVariables and not _activeInputs_->forceBinningUpdate ){
      contributionCount += sample.binning.isSmoothed ? _eventState_.contributions[event.resultIndex].size()
          : std::size_t(_eventState_.bins[event.resultIndex] >= 0);
      continue;
    }
    ++worker.rebinnedEvents;
    worker.changedSamples[event.samplePosition] = true;
    const auto& layout = sample.binning;
    const double* values = _eventState_.variables.data() + event.firstVariable;
    if( layout.isSmoothed ){
      for( std::size_t i = 0; i < layout.variableNames.size(); ++i ){
        LogThrowIf(not std::isfinite(values[i]), "Nonfinite binning variable: " << layout.variableNames[i]);
      }
    }
    const auto& positions = _reusableBinPositions_[event.samplePosition];
    const int previousBin = _eventState_.bins[event.resultIndex];
    if( previousBin >= 0 and std::size_t(previousBin) < positions.size()
        and positions[previousBin] != std::size_t(-1)
        and std::all_of(values, values + layout.variableNames.size(), [](double value_){ return std::isfinite(value_); }) ){
      ++worker.candidatesTested;
      const auto& bin = layout.bins[positions[previousBin]];
      if( BinningSemantics::containsStrictBin(bin, layout.intervals.data(), values) ){
        // The strict assignment is unchanged; keep its existing contribution.
        ++contributionCount;
        _validBinning_[event.resultIndex] = true;
        continue;
      }
    }
    _binningIndices_[event.samplePosition].findCandidates(values, worker.candidates);
    worker.candidatesTested += worker.candidates.size();
    auto& contributions = _eventState_.contributions[event.resultIndex];
    contributions.clear();
    int hardBin{-1};
    double total{0};
    for( auto position : worker.candidates ){
      const auto& bin = layout.bins[position];
      bool contains = hardBin < 0 and BinningSemantics::containsStrictBin(bin, layout.intervals.data(), values);
      if( contains ){ hardBin = bin.binIndex; }
      // Strict membership already gives the complete response. Smoothed bins
      // still need their fractional response, including outside the hard bin.
      double fraction = layout.isSmoothed ? BinningSemantics::fraction(layout, bin, values) : double(contains);
      if( fraction == 0 ){ continue; }
      contributions.push_back({bin.binIndex, fraction});
      total += fraction;
      if( not layout.isSmoothed ){ break; }
    }
    _eventState_.bins[event.resultIndex] = hardBin;
    if( layout.isSmoothed ){
      double tolerance = 64 * std::numeric_limits<double>::epsilon() * std::max(std::size_t(1), contributions.size());
      const auto finalized = BinningSemantics::finalizeBinningTotal(total, tolerance, std::isfinite(total));
      LogThrowIf(not finalized.isValid, "Total binning response exceeds one: " << total);
      if( total > 1 ){
        auto largest = std::max_element(contributions.begin(), contributions.end(), [](const auto& a_, const auto& b_){ return a_.fraction < b_.fraction; });
        largest->fraction -= finalized.excess;
        total = finalized.total;
      }
      if( total < 1 ){ contributions.push_back({-1, 1 - total}); }
    }
    contributionCount += contributions.size();
    _validBinning_[event.resultIndex] = true;
  }
  worker.contributionCount = contributionCount;
}

void Backends::CpuBackend::runEventStateJob(const std::string& jobName_) {
  std::fill(_workerErrors_.begin(), _workerErrors_.end(), std::exception_ptr{});
  _threadPool_.runJob(jobName_);
  for( const auto& error : _workerErrors_ ){ if( error ){ std::rethrow_exception(error); } }
}
