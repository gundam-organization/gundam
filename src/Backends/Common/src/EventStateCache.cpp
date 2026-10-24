#include "EventStateCache.h"

#include <algorithm>

void Backends::EventStateCache::build(const PropagationView& model_) {
  *this = EventStateCache();
  _model_ = &model_;
  _sampleRevisions_.assign(model_.samples.size(), 1);
  std::vector<std::size_t> variableDials(model_.variableDialIndices.begin(), model_.variableDialIndices.end());
  for( const auto& target : model_.variableResponses ){
    if( target.overrideDial != std::size_t(-1) ){ variableDials.push_back(target.overrideDial); }
  }
  auto unique = [](auto& values_){
    std::sort(values_.begin(), values_.end());
    values_.erase(std::unique(values_.begin(), values_.end()), values_.end());
  };
  unique(variableDials);
  for( auto index : variableDials ){
    const auto& dial = model_.dials.at(index);
    for( std::size_t i = dial.firstInput; i < dial.firstInput + dial.inputCount; ++i ){
      _parameterIndices_.push_back(model_.dialInputs.at(i).parameterIndex);
    }
    if( dial.type == BackendDialType::ExternalWeight or isTableDial(dial.type) ){ _externalBlockIndices_.push_back(dial.externalBlockIndex); }
  }
  unique(_parameterIndices_);
  unique(_externalBlockIndices_);
}

void Backends::EventStateCache::prepare(const PropagationInputs& inputs_) {
  _variableInputsChanged_ = true;
  if( not _valid_ ){ return; }
  if( not inputs_.variableInputRevisions.empty() and _variableInputRevisions_ != inputs_.variableInputRevisions ){ return; }
  for( std::size_t i = 0; i < _parameterIndices_.size(); ++i ){
    if( _parameters_[i] != inputs_.parameters.values[_parameterIndices_[i]] ){ return; }
  }
  for( std::size_t i = 0; i < _externalBlockIndices_.size(); ++i ){
    if( _externalGenerations_[i] != inputs_.externalWeights[_externalBlockIndices_[i]].generation ){ return; }
  }
  _variableInputsChanged_ = false;
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
  if( _variableInputsChanged_ ){
    _parameters_.resize(_parameterIndices_.size());
    for( std::size_t i = 0; i < _parameterIndices_.size(); ++i ){
      _parameters_[i] = inputs_.parameters.values[_parameterIndices_[i]];
    }
    _externalGenerations_.resize(_externalBlockIndices_.size());
    for( std::size_t i = 0; i < _externalBlockIndices_.size(); ++i ){
      _externalGenerations_[i] = inputs_.externalWeights[_externalBlockIndices_[i]].generation;
    }
    _variableInputRevisions_ = inputs_.variableInputRevisions;
  }
  _hostVariables_ = inputs_.hostVariables;
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
