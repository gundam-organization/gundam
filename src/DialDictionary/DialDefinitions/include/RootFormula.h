//
// Created by Adrien Blanchet on 10/04/2023.
//

#ifndef GUNDAM_ROOTFORMULA_H
#define GUNDAM_ROOTFORMULA_H

#include "DialBase.h"
#include "FormulaEvaluator.h"

#include <string>


class RootFormula : public DialBase {

public:
  RootFormula() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<RootFormula>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"RootFormula"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;
  [[nodiscard]] std::string getSummary() const override;

  void setFormulaStr(const std::string& formulaStr_);
  void setInputNames(const std::vector<std::string>& names_, const std::vector<std::string>& constantNames_ = {});
  [[nodiscard]] int getInputCount() const{ return _evaluator_.getInputCount(); }
  [[nodiscard]] const std::string& getFormulaStr() const{ return _evaluator_.getFormulaStr(); }
  [[nodiscard]] const std::vector<int>& getParameterInputIndices() const{ return _evaluator_.getParameterInputIndices(); }

  TFormula& getFormula(){ return _evaluator_.getFormula(); }
  const TFormula& getFormula() const{ return _evaluator_.getFormula(); }


private:
  FormulaEvaluator _evaluator_{};

};


#endif //GUNDAM_ROOTFORMULA_H
