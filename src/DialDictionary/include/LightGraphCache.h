//
// Created by Adrien Blanchet on 24/01/2023.
//

#ifndef GUNDAM_LIGHTGRAPHCACHE_H
#define GUNDAM_LIGHTGRAPHCACHE_H

#include "DialBaseCache.h"
#include "LightGraphHandler.h"


class LightGraphCache : public DialBaseCache, public LightGraphHandler {

public:
  LightGraphCache() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<LightGraphCache>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"LightGraphCache"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override { return this->evaluateGraph(input_); }

};


#endif //GUNDAM_LIGHTGRAPHCACHE_H
