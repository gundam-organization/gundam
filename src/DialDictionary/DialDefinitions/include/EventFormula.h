/// @file EventFormula.h
/// @brief Declares EventFormula, a formula dial bound to the variables of one
///        event.
/// @ingroup DialDictionary

#ifndef GUNDAM_EVENT_FORMULA_H
#define GUNDAM_EVENT_FORMULA_H

#include "DialBase.h"

#include <memory>
#include <string>
#include <vector>

class Event;

/// Binds a shared Formula dial to the loaded inputs of one event. Parameter
/// values still come from the collection's ordinary, shared DialInputBuffer.
///
/// A dial is a response function of one or more fit parameters (see
/// DialBase). A Formula collection whose `dialInputList` has `eventVariable`
/// entries keeps an EventFormula::Definition
/// (DialCollection::getEventFormulaDefinition()). For each event,
/// EventDialCache::buildReferenceCache() wraps the collection's RootFormula
/// in an EventFormula that stores the values of those event variables, read
/// once when the cache is built. evalResponse() then passes the formula its
/// inputs in `dialInputList` order, taking each from the event or from the
/// collection's input buffer.
///
/// Not supported by Cache::Manager, which disables itself for the run when
/// it finds one.
/// @ingroup DialDictionary
class EventFormula : public DialBase {
public:
  /// Where one formula input comes from. Exactly one index is set.
  struct Input {
    int parameterIndex{-1}; ///< Position in the collection's input buffer; -1 for an event variable.
    int eventIndex{-1}; ///< Position in Definition::eventVariables; -1 for a parameter.
  };
  /// The inputs of a formula that reads event variables, shared by all the
  /// EventFormula objects of a collection.
  struct Definition {
    std::vector<Input> inputs; ///< One entry per formula input, in `dialInputList` order.
    std::vector<std::string> eventVariables; ///< Distinct event-variable names, without brackets.
  };

  /// Stores the event-variable values of one event.
  /// @param formula_ The collection's RootFormula. Non-owning: it must
  ///        outlive this object.
  /// @param definition_ The collection's input definition.
  /// @param event_ The event to read the variables from.
  /// @param mirrorEdges_ Mirror edges applied to each event value
  ///        (DialCollection::getEventInputMirrorEdges()); a NaN lower edge
  ///        means no mirroring.
  /// @warning Throws if `formula_` is not a RootFormula, if the definition
  ///          is null or its input count differs from the formula's, or if
  ///          the event lacks one of the variables.
  EventFormula(const DialBase* formula_, std::shared_ptr<const Definition> definition_,
               const Event& event_, const DialInputBuffer::ParameterReference::MirrorEdges& mirrorEdges_);

  /// Copies this dial; the copy shares the formula and the definition.
  /// @return A new EventFormula with the same event values.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<EventFormula>(*this); }

  /// Gives the type name of this dial.
  /// @return `"EventFormula"`.
  [[nodiscard]] std::string getDialTypeName() const override { return "EventFormula"; }

  /// Evaluates the formula for this event.
  /// @param input_ The collection's input buffer, with the parameter values.
  /// @return The RootFormula response, with each input taken from the stored
  ///         event values or from `input_`.
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  /// Describes the dial.
  /// @return The formula's summary, then each event variable with its
  ///         stored value.
  [[nodiscard]] std::string getSummary() const override;

private:
  /// Owned by the source DialCollection, like the input buffer and supervisor.
  const DialBase* _formula_;
  /// Where each formula input comes from.
  std::shared_ptr<const Definition> _definition_;
  /// Value of each event variable of the definition, after mirroring.
  std::vector<double> _eventValues_;
};

#endif
