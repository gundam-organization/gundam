#include "EventFormula.h"
#include "RootFormula.h"
#include "Event.h"

#include "Logger.h"

#include <cmath>
#include <sstream>
#include <utility>

EventFormula::EventFormula(const DialBase* formula_, std::shared_ptr<const Definition> definition_,
                           const Event& event_, const DialInputBuffer::ParameterReference::MirrorEdges& mirrorEdges_)
    : _formula_(formula_), _definition_(std::move(definition_)) {
  const auto* formula = dynamic_cast<const RootFormula*>(_formula_);
  LogThrowIf(formula == nullptr or _definition_ == nullptr, "EventFormula requires a Formula and an input definition");
  LogThrowIf(formula->getInputCount() != _definition_->inputs.size(),
             "EventFormula: named input count mismatch");
  _eventValues_.reserve(_definition_->eventVariables.size());
  for( const auto& name : _definition_->eventVariables ){
    const auto& variables = event_.getVariables();
    LogThrowIf(variables.findVarIndex(name, false) < 0,
               "EventFormula: missing eventVariable \"" << name << "\" on event " << event_);
    double value = variables.fetchVariable(name).getVarAsDouble();
    if( not std::isnan(mirrorEdges_.minValue) ){
      value = std::abs(std::fmod(value - mirrorEdges_.minValue, 2 * mirrorEdges_.range));
      if( value > mirrorEdges_.range ){ value = 2 * mirrorEdges_.range - value; }
      value += mirrorEdges_.minValue;
    }
    _eventValues_.emplace_back(value);
  }
}

DialInputBuffer EventFormula::bindInputs(const DialInputBuffer& input_) const {
  std::vector<double> values;
  values.reserve(_definition_->inputs.size());
  for( const auto& source : _definition_->inputs ){
    values.emplace_back(source.eventIndex >= 0 ? _eventValues_[source.eventIndex]
                                             : input_.getInputBuffer().at(source.parameterIndex));
  }
  DialInputBuffer buffer;
  buffer.setInputValues(std::move(values));
  return buffer;
}

double EventFormula::evalResponse(const DialInputBuffer& input_) const {
  return _formula_->evalResponse(bindInputs(input_));
}

bool EventFormula::hasAnalyticGradient() const {
  return _formula_->hasAnalyticGradient();
}

double EventFormula::evalGradient(const DialInputBuffer& input_, int iInput_) const {
  if( iInput_ < 0 or size_t(iInput_) >= input_.getInputBuffer().size() ){ return 0.; }
  const auto buffer = bindInputs(input_);
  double gradient{0.};
  // The caller indexes the collection's parameter-only input buffer. Map it
  // into the formula's mixed layout and keep loaded event values constant.
  for( size_t i = 0; i < _definition_->inputs.size(); ++i ){
    const auto& source = _definition_->inputs[i];
    if( source.eventIndex < 0 and source.parameterIndex == iInput_ ){
      gradient += _formula_->evalGradient(buffer, int(i));
    }
  }
  return gradient;
}

std::string EventFormula::getSummary() const {
  std::stringstream ss;
  ss << _formula_->getSummary() << ", loaded event inputs: ";
  for( size_t i = 0 ; i < _eventValues_.size() ; ++i ){
    if( i != 0 ){ ss << ", "; }
    ss << _definition_->eventVariables[i] << "=" << _eventValues_[i];
  }
  return ss.str();
}
