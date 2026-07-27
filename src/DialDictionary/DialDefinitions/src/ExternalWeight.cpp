#include "ExternalWeight.h"

#include "Logger.h"

double ExternalWeight::evalResponse(const DialInputBuffer& input_) const {
  LogThrowIf(_weightList_ == nullptr, "ExternalWeight has no weight buffer attached.");
  LogThrowIf(_eventIndex_ >= _weightList_->size(),
             "ExternalWeight event index out of range: " << _eventIndex_ << " >= " << _weightList_->size());
  return _weightList_->at(_eventIndex_);
}

std::string ExternalWeight::getSummary() const {
  return "ExternalWeight eventIndex=" + std::to_string(_eventIndex_);
}
