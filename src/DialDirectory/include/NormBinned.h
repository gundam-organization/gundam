//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_NORMBINNED_H
#define GUNDAM_NORMBINNED_H

#include "Norm.h"
#include "DialBinned.h"


class NormBinned : public Norm, public DialBinned {

public:
  NormBinned() = default;
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<NormBinned>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"NormBinned"}; }

};


#endif //GUNDAM_NORMBINNED_H
