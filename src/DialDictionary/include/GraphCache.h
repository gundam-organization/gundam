//
// Created by Adrien Blanchet on 07/12/2022.
//

#ifndef GUNDAM_GRAPHCACHE_H
#define GUNDAM_GRAPHCACHE_H

#include "DialBaseCache.h"
#include "GraphHandler.h"


class GraphCache : public DialBaseCache, public GraphHandler {

public:
  GraphCache() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<GraphCache>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"Graph"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override { return this->evaluateGraph(input_); }

};


#endif //GUNDAM_GRAPHCACHE_H
