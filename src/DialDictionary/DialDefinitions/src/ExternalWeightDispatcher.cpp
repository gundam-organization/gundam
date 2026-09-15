#include "ExternalWeightDispatcher.h"

#include "Logger.h"

double ExternalWeightDispatcher::evalResponse(const DialInputBuffer& input_) const {
  LogThrowIf(_weightSource_ == nullptr, "ExternalWeightDispatcher has no weight buffer attached.");
  LogThrowIf(_eventIndex_ >= _weightSource_->size(),
             "ExternalWeightDispatcher event index out of range: " << _eventIndex_ << " >= " << _weightSource_->size());
  return _weightSource_->data() == nullptr ? 1. : _weightSource_->data()[_eventIndex_];
}

std::string ExternalWeightDispatcher::getSummary() const {
  return "ExternalWeightDispatcher eventIndex=" + std::to_string(_eventIndex_);
}
