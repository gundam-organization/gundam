//
// Created by Adrien Blanchet on 10/04/2023.
//

#ifndef GUNDAM_ROOTFORMULA_H
#define GUNDAM_ROOTFORMULA_H

#include "DialBase.h"

#include "TFormula.h"

#include <string>


class RootFormula : public DialBase {

public:
  RootFormula() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<RootFormula>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"RootFormula"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;
  [[nodiscard]] std::string getSummary() const override;

  void setFormulaStr(const std::string& formulaStr_);


private:
  TFormula _formula_{};

};


#endif //GUNDAM_ROOTFORMULA_H
