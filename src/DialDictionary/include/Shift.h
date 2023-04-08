//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_NORM_H
#define GUNDAM_NORM_H

#include "DialBase.h"

// for norm dial, no cache is needed


class Shift : public DialBase {

public:
  Shift() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Shift>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"Shift"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override { return _shiftValue_; }

  /// Build the dial with no input arguments.  This is here for completeness,
  /// but could eventually do... something.
  virtual void buildDial(std::string option="") override {}

  void setShiftValue(double shiftValue_) { _shiftValue_ = shiftValue_; }

private:
  double _shiftValue_{1};

};


#endif //GUNDAM_NORM_H
