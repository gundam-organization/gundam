//
// Created by Adrien Blanchet on 10/04/2023.
//

#ifndef GUNDAM_POLYNOMIAL_H
#define GUNDAM_POLYNOMIAL_H

#include "DialBase.h"

#include "vector"


class Polynomial : public DialBase {

public:
  Polynomial() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Polynomial>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"Polynomial"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  // Initialize the polynomial ocoefficients.  The coefficients go in the first
  // vector, and the other two are ignored.
  virtual void buildDial(const std::vector<double>& coefficientList_,
                         const std::vector<double>&,
                         const std::vector<double>&,
                         const std::string& option_="") override;

  void setCoefficientList(const std::vector<double> &coefficientList_);


private:
  std::vector<double> _coefficientList_{};

};


#endif //GUNDAM_POLYNOMIAL_H
