#ifndef GUNDAM_EVENT_FORMULA_H
#define GUNDAM_EVENT_FORMULA_H

#include "DialBase.h"

#include <memory>
#include <string>
#include <vector>

class Event;

// Binds a shared Formula dial to the loaded inputs of one event. Parameter
// values still come from the collection's ordinary, shared DialInputBuffer.
class EventFormula : public DialBase {
public:
  struct Input {
    int parameterIndex{-1};
    int eventIndex{-1};
  };
  struct Definition {
    std::vector<Input> inputs;
    std::vector<std::string> eventVariables;
  };

  EventFormula(const DialBase* formula_, std::shared_ptr<const Definition> definition_,
               const Event& event_, const DialInputBuffer::ParameterReference::MirrorEdges& mirrorEdges_);
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<EventFormula>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return "EventFormula"; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;
  [[nodiscard]] std::string getSummary() const override;

private:
  // Owned by the source DialCollection, like the input buffer and supervisor.
  const DialBase* _formula_;
  std::shared_ptr<const Definition> _definition_;
  std::vector<double> _eventValues_;
};

#endif
