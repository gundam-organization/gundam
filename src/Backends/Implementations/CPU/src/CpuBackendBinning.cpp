#include "CpuBackend.h"

#include "Semantics/BackendDialSemantics.h"

#include "Logger.h"

#include <algorithm>
#include <cmath>
#include <limits>

double Backends::CpuBackend::evalResponse(std::size_t dial_) const {
  return _isCachedDial_[dial_] ? _cachedDialResponses_[dial_]
      : Semantics::evalDialResponse(_engineView_.propagation, _engineView_.propagation.dials[dial_], *_activeInputs_);
}

void Backends::CpuBackend::updateEventState(const PropagationInputs& inputs_) {
  const auto& model = _engineView_.propagation;
  _activeInputs_ = &inputs_;
  _variableInputsChanged_ = not _eventStateValid_;
  for( std::size_t iDial = 0; iDial < model.dials.size(); ++iDial ){
    if( not _variableDialDependencies_[iDial] ){ continue; }
    const auto& dial = model.dials[iDial];
    if( not inputs_.dialInputRevisions.empty()
        and (_lastDialRevisions_.size() != inputs_.dialInputRevisions.size()
             or _lastDialRevisions_[iDial] != inputs_.dialInputRevisions[iDial]) ){
      _variableInputsChanged_ = true;
    }
    for( std::size_t i = dial.firstInput; i < dial.firstInput + dial.inputCount; ++i ){
      auto parameter = model.dialInputs[i].parameterIndex;
      if( _lastBinningParameters_.size() != inputs_.parameters.values.size()
          or _lastBinningParameters_[parameter] != inputs_.parameters.values[parameter] ){
        _variableInputsChanged_ = true;
      }
    }
    if( dial.type == BackendDialType::ExternalWeight ){
      auto block = dial.externalBlockIndex;
      if( _lastExternalGenerations_.size() != inputs_.externalWeights.size()
          or _lastExternalGenerations_[block] != inputs_.externalWeights[block].generation ){
        _variableInputsChanged_ = true;
      }
    }
  }
  if( not _variableInputsChanged_ and inputs_.hostVariables.empty() and not inputs_.forceBinningUpdate ){
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
  }
  for( const auto& contributions : _eventState_.contributions ){ _lastTiming_.binContributionCount += contributions.size(); }
  _cachedContributionCount_ = _lastTiming_.binContributionCount;
  if( _variableInputsChanged_ ){ ++_eventState_.variableRevision; }
  for( std::size_t i = 0; i < model.samples.size(); ++i ){
    bool changed = _variableInputsChanged_ and model.samples[i].useDynamicBinning;
    for( const auto& worker : _binningWorkers_ ){ changed = changed or worker.changedSamples[i]; }
    if( changed ){ ++_eventState_.sampleRevisions[i]; }
  }
  if( _variableInputsChanged_ or _lastTiming_.rebinnedEventCount != 0 ){ ++_eventState_.revision; }
  // Commit only after both stages complete successfully.
  _lastBinningParameters_ = inputs_.parameters.values;
  _lastExternalGenerations_.clear();
  for( const auto& block : inputs_.externalWeights ){ _lastExternalGenerations_.push_back(block.generation); }
  _lastHostVariables_ = inputs_.hostVariables;
  _lastDialRevisions_ = inputs_.dialInputRevisions;
  _eventStateValid_ = true;
  _activeInputs_ = nullptr;
}

void Backends::CpuBackend::updateEventVariablesThread(int iThread_) {
  const auto& model = _engineView_.propagation;
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
        if( inputs.forceBinningUpdate or _lastHostVariables_.size() != inputs.hostVariables.size()
            or inputs.hostVariables[i] != _lastHostVariables_[i] ){ assign(i, inputs.hostVariables[i]); }
      }
    }
    if( not _variableInputsChanged_ ){ continue; }
    for( std::size_t i = event.firstVariableResponse; i < event.firstVariableResponse + event.variableResponseCount; ++i ){
      const auto& target = model.variableResponses[i];
      double value = Semantics::evalVariableResponse(target, model.variableDialIndices.data(),
                                                      [this](std::size_t dial_){ return evalResponse(dial_); });
      assign(target.variableIndex, value);
    }
  }
}

void Backends::CpuBackend::updateEventBinningThread(int iThread_) {
  const auto& model = _engineView_.propagation;
  auto& worker = _binningWorkers_[iThread_];
  worker.rebinnedEvents = worker.candidatesTested = 0;
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
    if( _validBinning_[event.resultIndex] and not sample.readHostVariables and not _activeInputs_->forceBinningUpdate ){ continue; }
    ++worker.rebinnedEvents;
    worker.changedSamples[event.samplePosition] = true;
    const auto& layout = sample.binning;
    const double* values = _eventState_.variables.data() + event.firstVariable;
    if( layout.isSmoothed ){
      for( std::size_t i = 0; i < layout.variableNames.size(); ++i ){
        LogThrowIf(not std::isfinite(values[i]), "Nonfinite binning variable: " << layout.variableNames[i]);
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
      if( hardBin < 0 ){
        bool contains{true};
        for( std::size_t i = bin.firstInterval; i < bin.firstInterval + bin.intervalCount; ++i ){
          const auto& interval = layout.intervals[i];
          double value = values[interval.variableIndex];
          if( interval.lowerBound == interval.upperBound ? value != interval.lowerBound
              : value < interval.lowerBound or value >= interval.upperBound ){ contains = false; break; }
        }
        if( contains ){ hardBin = bin.binIndex; }
      }
      double fraction = BinningSemantics::fraction(layout, bin, values);
      if( fraction == 0 ){ continue; }
      contributions.push_back({bin.binIndex, fraction});
      total += fraction;
      if( not layout.isSmoothed ){ break; }
    }
    _eventState_.bins[event.resultIndex] = hardBin;
    if( layout.isSmoothed ){
      double tolerance = 64 * std::numeric_limits<double>::epsilon() * std::max(std::size_t(1), contributions.size());
      LogThrowIf(not std::isfinite(total) or total > 1 + tolerance, "Total binning response exceeds one: " << total);
      if( total > 1 ){
        auto largest = std::max_element(contributions.begin(), contributions.end(), [](const auto& a_, const auto& b_){ return a_.fraction < b_.fraction; });
        largest->fraction -= total - 1;
        total = 1;
      }
      if( total < 1 ){ contributions.push_back({-1, 1 - total}); }
    }
    _validBinning_[event.resultIndex] = true;
  }
}

void Backends::CpuBackend::runEventStateJob(const std::string& jobName_) {
  std::fill(_workerErrors_.begin(), _workerErrors_.end(), std::exception_ptr{});
  _threadPool_.runJob(jobName_);
  for( const auto& error : _workerErrors_ ){ if( error ){ std::rethrow_exception(error); } }
}
