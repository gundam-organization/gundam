//
// Created by Adrien Blanchet on 22/01/2023.
//

#ifndef GUNDAM_MONOTONICSPLINECACHE_H
#define GUNDAM_MONOTONICSPLINECACHE_H

#include "DialBaseCache.h"
#include "MonotonicSplineHandler.h"

class MonotonicSplineCache : public DialBaseCache, public MonotonicSplineHandler {

public:
  MonotonicSplineCache() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<MonotonicSplineCache>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"MonotonicSplineCache"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override { return this->evaluateSpline(input_); }

};


#endif //GUNDAM_MONOTONICSPLINECACHE_H
