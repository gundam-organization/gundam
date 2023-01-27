//
// Created by Adrien Blanchet on 24/01/2023.
//

#ifndef GUNDAM_SIMPLESPLINE_H
#define GUNDAM_SIMPLESPLINE_H

#include "DialBase.h"
#include "SimpleSplineHandler.h"

class SimpleSpline : public DialBase, public SimpleSplineHandler {

public:
  SimpleSpline() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<SimpleSpline>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"SimpleSpline"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override { return this->evaluateSpline(input_); }

};


#endif //GUNDAM_SIMPLESPLINE_H
