//
// Created by Adrien Blanchet on 23/01/2023.
//

#ifndef GUNDAM_UNIFORMSPLINE_H
#define GUNDAM_UNIFORMSPLINE_H

#include "DialBase.h"
#include "UniformSplineHandler.h"

class UniformSpline : public DialBase, public UniformSplineHandler {

public:
  UniformSpline() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<UniformSpline>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"UniformSpline"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override { return this->evaluateSpline(input_); }

};


#endif //GUNDAM_UNIFORMSPLINE_H
