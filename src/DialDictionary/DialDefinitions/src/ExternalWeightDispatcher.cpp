/// @file ExternalWeightDispatcher.cpp
/// @brief Implements ExternalWeightDispatcher: reading one external weight.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in
/// ExternalWeightDispatcher.h. The comments here describe how each method
/// does its job.

#include "ExternalWeightDispatcher.h"

#include "Logger.h"

/// @par Implementation
/// Checks the list and the index on every call, then reads the entry with
/// `at()`.
double ExternalWeightDispatcher::evalResponse(const DialInputBuffer& input_) const {
  LogThrowIf(_weightList_ == nullptr, "ExternalWeightDispatcher has no weight buffer attached.");
  LogThrowIf(_eventIndex_ >= _weightList_->size(),
             "ExternalWeightDispatcher event index out of range: " << _eventIndex_ << " >= " << _weightList_->size());
  return _weightList_->at(_eventIndex_);
}

/// @par Implementation
/// Concatenates the type name and the index.
std::string ExternalWeightDispatcher::getSummary() const {
  return "ExternalWeightDispatcher eventIndex=" + std::to_string(_eventIndex_);
}
