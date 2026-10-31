#ifndef GUNDAM_EXTERNAL_WEIGHT_DISPATCHER_H
#define GUNDAM_EXTERNAL_WEIGHT_DISPATCHER_H

#include "DialBase.h"
#include "ExternalWeightBuffer.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

class ExternalWeightDispatcher : public DialBase {

public:
  ExternalWeightDispatcher() = delete;
  ExternalWeightDispatcher(std::shared_ptr<const ExternalWeightBuffer> weightSource_, std::size_t eventIndex_)
      : _weightSource_(std::move(weightSource_)), _eventIndex_(eventIndex_) {}

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<ExternalWeightDispatcher>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"ExternalWeightDispatcher"}; }

  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;
  [[nodiscard]] std::string getSummary() const override;

  [[nodiscard]] const std::shared_ptr<const ExternalWeightBuffer>& getWeightSource() const { return _weightSource_; }
  [[nodiscard]] std::size_t getWeightIndex() const { return _eventIndex_; }

private:
  std::shared_ptr<const ExternalWeightBuffer> _weightSource_{nullptr};
  std::size_t _eventIndex_{std::size_t(-1)};

};

#endif // GUNDAM_EXTERNAL_WEIGHT_DISPATCHER_H
