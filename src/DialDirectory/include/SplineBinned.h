//
// Created by Adrien Blanchet on 30/11/2022.
//

#ifndef GUNDAM_SPLINEBINNED_H
#define GUNDAM_SPLINEBINNED_H

#include "DialBase.h"
#include "DialSplineHandler.h"
#include "DialBinned.h"



class SplineBinned : public DialBase, public DialSplineHandler, public DialBinned {

public:
  SplineBinned() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<SplineBinned>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"SplineBinned"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override;

};


#endif //GUNDAM_SPLINEBINNED_H
