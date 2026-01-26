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

  // mandatory overrides
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Norm>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"Norm"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override { return input_.getInputBuffer()[0]; }

};


#endif //GUNDAM_NORM_H
