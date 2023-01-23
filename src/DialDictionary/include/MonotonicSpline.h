//
// Created by Adrien Blanchet on 22/01/2023.
//

#ifndef GUNDAM_MONOTONICSPLINE_H
#define GUNDAM_MONOTONICSPLINE_H

#include "DialBase.h"
#include "MonotonicSplineHandler.h"

class MonotonicSpline : public DialBase, public MonotonicSplineHandler {

public:
  MonotonicSpline() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<MonotonicSpline>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"MonotonicSpline"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override { return this->evaluateSpline(input_); }

};


#endif //GUNDAM_MONOTONICSPLINE_H
