#ifndef GUNDAM_EXTERNAL_WEIGHT_H
#define GUNDAM_EXTERNAL_WEIGHT_H

#include "DialBase.h"

#include <string>
#include <vector>

class ExternalWeight : public DialBase {

public:
  ExternalWeight() = delete;
  ExternalWeight(const std::vector<double>* weightList_, std::size_t eventIndex_)
      : _weightList_(weightList_), _eventIndex_(eventIndex_) {}

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<ExternalWeight>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"ExternalWeight"}; }

  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;
  [[nodiscard]] std::string getSummary() const override;

private:
  const std::vector<double>* _weightList_{nullptr};
  std::size_t _eventIndex_{std::size_t(-1)};

};

#endif // GUNDAM_EXTERNAL_WEIGHT_H
