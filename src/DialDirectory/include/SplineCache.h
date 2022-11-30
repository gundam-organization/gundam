//
// Created by Adrien Blanchet on 30/11/2022.
//

#ifndef GUNDAM_SPLINECACHE_H
#define GUNDAM_SPLINECACHE_H

#include "DialBaseCache.h"
#include "DialSplineHandler.h"


class SplineCache : public DialBaseCache, public  DialSplineHandler {

public:
  SplineCache() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<SplineCache>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"SplineCache"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override;

};


#endif //GUNDAM_SPLINECACHE_H
