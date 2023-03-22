//
// Created by Adrien Blanchet on 22/01/2023.
//

#ifndef GUNDAM_COMPACTSPLINE_H
#define GUNDAM_COMPACTSPLINE_H

#include "DialBase.h"
#include "CompactSplineHandler.h"

class CompactSpline : public DialBase, public CompactSplineHandler {

public:
  CompactSpline() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<CompactSpline>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"CompactSpline"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override { return this->evaluateSpline(input_); }

};


#endif //GUNDAM_COMPACTSPLINE_H
