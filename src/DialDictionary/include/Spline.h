//
// Created by Adrien Blanchet on 30/11/2022.
//

#ifndef GUNDAM_SPLINE_H
#define GUNDAM_SPLINE_H

#include "DialBase.h"
#include "SplineHandler.h"

class Spline : public DialBase, public SplineHandler {

public:
  Spline() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Spline>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"Spline"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override { return this->evaluateSpline(input_); }

};


#endif //GUNDAM_SPLINE_H
