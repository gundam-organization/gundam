/// @file EventFormula.cpp
/// @brief Implements EventFormula: reading the event variables and
///        evaluating the formula.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in EventFormula.h. The
/// comments here describe how each method does its job.

#include "EventFormula.h"
#include "RootFormula.h"
#include "Event.h"

#include "Logger.h"

#include <cmath>
#include <sstream>
#include <utility>

/// @par Implementation
/// Reads each event variable as a double and applies the same
/// triangle-wave mirroring as DialInputBuffer::update() when a lower edge is
/// set. The values are read once, here, and not updated afterwards.
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

/// @par Implementation
/// Builds the formula inputs in a local DialInputBuffer
/// (DialInputBuffer::setInputValues()), so several threads may evaluate the
/// same EventFormula at once.
double EventFormula::evalResponse(const DialInputBuffer& input_) const {
  std::vector<double> values;
  values.reserve(_definition_->inputs.size());
  for( const auto& source : _definition_->inputs ){
    values.emplace_back(source.eventIndex >= 0 ? _eventValues_[source.eventIndex]
                                             : input_.getInputBuffer().at(source.parameterIndex));
  }
  DialInputBuffer buffer;
  buffer.setInputValues(std::move(values));
  return _formula_->evalResponse(buffer);
}

/// @par Implementation
/// Appends `name=value` for each event variable to the formula's summary.
std::string EventFormula::getSummary() const {
  std::stringstream ss;
  ss << _formula_->getSummary() << ", loaded event inputs: ";
  for( size_t i = 0 ; i < _eventValues_.size() ; ++i ){
    if( i != 0 ){ ss << ", "; }
    ss << _definition_->eventVariables[i] << "=" << _eventValues_[i];
  }
  return ss.str();
}
