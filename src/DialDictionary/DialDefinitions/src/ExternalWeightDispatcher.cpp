#include "ExternalWeightDispatcher.h"

#include "Logger.h"

bool ExternalWeightDispatcher::hasAnalyticGradient() const {
  if( _weightSource_ == nullptr or not _weightSource_->supportsGradient() ){ return false; }
  _weightSource_->requestGradient();
  return true;
}

std::vector<double> ExternalWeightDispatcher::evalGradients(const DialInputBuffer& input_) const {
  LogThrowIf(_weightSource_ == nullptr or not _weightSource_->supportsGradient(),
             "ExternalWeightDispatcher: worker does not provide analytic gradients.");
  LogThrowIf(not _weightSource_->hasCurrentGradient(),
             "ExternalWeightDispatcher: gradient is missing or stale; propagate after requesting gradients.");
  const auto count = _weightSource_->getGradientInputCount();
  LogThrowIf(input_.getInputBuffer().size() != count, "ExternalWeightDispatcher: gradient input count mismatch.");
  LogThrowIf(input_.getInputBuffer() != _weightSource_->getInputValues(),
             "ExternalWeightDispatcher: gradient does not match the supplied input point.");
  LogThrowIf(_eventIndex_ >= _weightSource_->size(), "ExternalWeightDispatcher: gradient event index out of range.");
  if( count == 0 ){ return {}; }
  const auto* row = _weightSource_->gradientData() + _eventIndex_ * count;
  return {row, row + count};
}

double ExternalWeightDispatcher::evalGradient(const DialInputBuffer& input_, int iInput_) const {
  LogThrowIf(iInput_ < 0 or size_t(iInput_) >= input_.getInputBuffer().size(),
             "ExternalWeightDispatcher: invalid gradient input index " << iInput_);
  return evalGradients(input_)[iInput_];
}

double ExternalWeightDispatcher::evalResponse(const DialInputBuffer& input_) const {
  LogThrowIf(_weightSource_ == nullptr, "ExternalWeightDispatcher has no weight buffer attached.");
  LogThrowIf(_eventIndex_ >= _weightSource_->size(),
             "ExternalWeightDispatcher event index out of range: " << _eventIndex_ << " >= " << _weightSource_->size());
  return _weightSource_->data() == nullptr ? 1. : _weightSource_->data()[_eventIndex_];
}

std::string ExternalWeightDispatcher::getSummary() const {
  return "ExternalWeightDispatcher eventIndex=" + std::to_string(_eventIndex_);
}
