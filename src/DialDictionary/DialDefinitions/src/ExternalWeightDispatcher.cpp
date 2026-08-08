#include "ExternalWeightDispatcher.h"

#include "Logger.h"

double ExternalWeightDispatcher::evalResponse(const DialInputBuffer& input_) const {
  LogThrowIf(_weightList_ == nullptr, "ExternalWeightDispatcher has no weight buffer attached.");
  LogThrowIf(_eventIndex_ >= _weightList_->size(),
             "ExternalWeightDispatcher event index out of range: " << _eventIndex_ << " >= " << _weightList_->size());
  return _weightList_->at(_eventIndex_);
}

std::string ExternalWeightDispatcher::getSummary() const {
  return "ExternalWeightDispatcher eventIndex=" + std::to_string(_eventIndex_);
}
