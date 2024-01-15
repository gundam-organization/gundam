//
// Created by Nadrino on 29/11/2022.
//

#ifndef GUNDAM_SHIFT_H
#define GUNDAM_SHIFT_H

#include "DialBase.h"

// Implement a dial to apply a scale factor (not a variable shift).  This can
// be used to replace a Spline or a Graph when it corresponds to a constant
// value.
class Shift : public DialBase {

public:
  Shift() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Shift>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"Shift"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override { return _shiftValue_; }

  void buildDial(double shift_, const std::string& options_="") override { _shiftValue_ = shift_; }

private:
  double _shiftValue_{1};

};
#endif //GUNDAM_SHIFT_H
