//
// Created by Adrien Blanchet on 10/04/2023.
//

#ifndef GUNDAM_POLYNOMIAL_H
#define GUNDAM_POLYNOMIAL_H

#include "DialBase.h"
#include "DialUtils.h"

#include <vector>


class Polynomial : public DialBase {

public:
  Polynomial() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Polynomial>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"Polynomial"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  void setAllowExtrapolation(bool allowExtrapolation_) override { _allowExtrapolation_ = allowExtrapolation_; }

  void setCoefficientList(const std::vector<double> &coefficientList_){ _coefficientList_ = coefficientList_; }
  void setSplineBounds(const DialUtils::Range& splineBounds_){ _splineBounds_ = splineBounds_; }

private:
  std::vector<double> _coefficientList_{};
  DialUtils::Range _splineBounds_{std::nan("unset"), std::nan("unset")};
  bool _allowExtrapolation_{false};

};


#endif //GUNDAM_POLYNOMIAL_H
