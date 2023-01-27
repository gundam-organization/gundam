//
// Created by Adrien Blanchet on 24/01/2023.
//

#ifndef GUNDAM_SIMPLESPLINECACHE_H
#define GUNDAM_SIMPLESPLINECACHE_H


#include "DialBaseCache.h"
#include "SimpleSplineHandler.h"


class SimpleSplineCache : public DialBaseCache, public SimpleSplineHandler {

public:
  SimpleSplineCache() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<SimpleSplineCache>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"SimpleSplineCache"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override { return this->evaluateSpline(input_); }

};


#endif //GUNDAM_SIMPLESPLINECACHE_H
