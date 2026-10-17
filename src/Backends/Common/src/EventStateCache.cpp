#include "EventStateCache.h"

#include <algorithm>

void Backends::EventStateCache::build(const PropagationView& model_) {
  *this = EventStateCache();
  _model_ = &model_;
  _sampleRevisions_.assign(model_.samples.size(), 1);
  for( auto dial : model_.variableDialIndices ){ _variableDials_.push_back(dial); }
  for( const auto& target : model_.variableResponses ){
    if( target.overrideDial != std::size_t(-1) ){ _variableDials_.push_back(target.overrideDial); }
  }
  std::sort(_variableDials_.begin(), _variableDials_.end());
  _variableDials_.erase(std::unique(_variableDials_.begin(), _variableDials_.end()), _variableDials_.end());
}

void Backends::EventStateCache::prepare(const PropagationInputs& inputs_) {
  _variableInputsChanged_ = not _valid_;
  for( auto index : _variableDials_ ){
    const auto& dial = _model_->dials.at(index);
    if( not inputs_.dialInputRevisions.empty()
        and (_dialRevisions_.size() != inputs_.dialInputRevisions.size() or _dialRevisions_[index] != inputs_.dialInputRevisions[index]) ){
      _variableInputsChanged_ = true;
    }
    for( std::size_t i = dial.firstInput; i < dial.firstInput + dial.inputCount; ++i ){
      auto parameter = _model_->dialInputs[i].parameterIndex;
      if( _parameters_.size() != inputs_.parameters.values.size() or _parameters_[parameter] != inputs_.parameters.values[parameter] ){
        _variableInputsChanged_ = true;
      }
    }
    if( dial.type == BackendDialType::ExternalWeight ){
      auto block = dial.externalBlockIndex;
      if( _externalGenerations_.size() != inputs_.externalWeights.size() or _externalGenerations_[block] != inputs_.externalWeights[block].generation ){
        _variableInputsChanged_ = true;
      }
    }
  }
}

bool Backends::EventStateCache::needsUpdate(const PropagationInputs& inputs_) const {
  return _variableInputsChanged_ or not inputs_.hostVariables.empty() or inputs_.forceBinningUpdate;
}

bool Backends::EventStateCache::hostVariableChanged(std::size_t index_, const PropagationInputs& inputs_) const {
  return inputs_.forceBinningUpdate or _hostVariables_.size() != inputs_.hostVariables.size()
      or _hostVariables_[index_] != inputs_.hostVariables[index_];
}

void Backends::EventStateCache::commit(const PropagationInputs& inputs_, const std::vector<std::uint8_t>& changedSamples_) {
  bool anyChanged{false};
  for( std::size_t i = 0; i < _model_->samples.size(); ++i ){
    if( changedSamples_.at(i) or (_variableInputsChanged_ and _model_->samples[i].useDynamicBinning) ){
      ++_sampleRevisions_[i];
      anyChanged = true;
    }
  }
  if( _variableInputsChanged_ ){ ++_variableRevision_; }
  if( anyChanged or _variableInputsChanged_ ){ ++_revision_; }
  _parameters_ = inputs_.parameters.values;
  _hostVariables_ = inputs_.hostVariables;
  _dialRevisions_ = inputs_.dialInputRevisions;
  _externalGenerations_.clear();
  for( const auto& block : inputs_.externalWeights ){ _externalGenerations_.push_back(block.generation); }
  _valid_ = true;
}

void Backends::EventStateCache::publishRevisions(EventState& state_) const {
  state_.revision = _revision_;
  state_.variableRevision = _variableRevision_;
  state_.sampleRevisions = _sampleRevisions_;
}

void Backends::EventStateCache::invalidate() {
  _valid_ = false;
  _hostVariables_.clear();
}
