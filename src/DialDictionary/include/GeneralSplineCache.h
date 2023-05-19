//
// Created by Adrien Blanchet on 23/01/2023.
//

#ifndef GUNDAM_GENERALSPLINECACHE_H
#define GUNDAM_GENERALSPLINECACHE_H

#include "DialBaseCache.h"
#include "GeneralSplineHandler.h"


class GeneralSplineCache : public DialBaseCache, public GeneralSplineHandler {

public:
  GeneralSplineCache() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<GeneralSplineCache>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"GeneralSplineCache"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override { return this->evaluateSpline(input_); }

};


#endif //GUNDAM_GENERALSPLINECACHE_H
