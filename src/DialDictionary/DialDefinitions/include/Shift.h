//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_SHIFT_H
#define GUNDAM_SHIFT_H

#include "DialBase.h"

// for norm dial, no cache is needed


class Shift : public DialBase {

public:
  Shift() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Shift>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"Shift"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override { return _shiftValue_; }

  void buildDial(void* shiftValuePtr_, const std::string& options_="") override { _shiftValue_ = *((double*) shiftValuePtr_); }

private:
  double _shiftValue_{1};

};


#endif //GUNDAM_SHIFT_H
