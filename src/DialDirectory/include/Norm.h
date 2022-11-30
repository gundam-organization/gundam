//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_NORM_H
#define GUNDAM_NORM_H

#include "DialBase.h"

// for norm dial, no cache is needed


class Norm : public DialBase {

public:
  Norm() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Norm>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"Norm"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override;

};


#endif //GUNDAM_NORM_H
