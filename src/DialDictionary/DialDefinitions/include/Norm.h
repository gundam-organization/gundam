//
// Created by Nadrino on 29/11/2022.
//

#ifndef GUNDAM_NORM_H
#define GUNDAM_NORM_H

#include "DialBase.h"

// for norm dial, no cache is needed


class Norm : public DialBase {

public:
  Norm() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Norm>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"Norm"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override { return input_.getBuffer()[0]; }

  /// Build the dial with no input arguments.  This is here for completeness,
  /// but could eventually do... something.
  virtual void buildDial(const std::string& option_="") override {}

};


#endif //GUNDAM_NORM_H
