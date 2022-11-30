//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_SPLINECACHEBINNED_H
#define GUNDAM_SPLINECACHEBINNED_H

#include "SplineCache.h"
#include "DialBinned.h"

class SplineCacheBinned : public SplineCache, public DialBinned {

public:
  SplineCacheBinned() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<SplineCacheBinned>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"SplineCacheBinned"}; }

};


#endif //GUNDAM_SPLINECACHEBINNED_H
