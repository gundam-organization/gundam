//
// Created by Adrien Blanchet on 23/01/2023.
//

#ifndef GUNDAM_GENERALSPLINE_H
#define GUNDAM_GENERALSPLINE_H

#include "DialBase.h"
#include "GeneralSplineHandler.h"

class GeneralSpline : public DialBase, public GeneralSplineHandler {

public:
  GeneralSpline() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<GeneralSpline>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"GeneralSpline"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override { return this->evaluateSpline(input_); }

};


#endif //GUNDAM_GENERALSPLINE_H
