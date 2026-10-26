//
// Created by Adrien Blanchet on 10/04/2023.
//

#include "RootFormula.h"


double RootFormula::evalResponse(const DialInputBuffer& input_) const {
  return _formula_.EvalPar(&input_.getInputBuffer()[0]);
}


void RootFormula::setFormulaStr(const std::string& formulaStr_){
  // Dial formulas are independent. Registering an expression such as x[0]
  // as a global formula name can make later nodes recursively expand it.
  _formula_ = TFormula(formulaStr_.c_str(), formulaStr_.c_str(), false);
  LogThrowIf(not _formula_.IsValid(), "\"" << formulaStr_ << "\": could not be parsed as formula expression.");
}

std::string RootFormula::getSummary() const {
  std::stringstream ss;
  ss << this->getDialTypeName() << ": formula: " << _formula_.GetExpFormula();
  return ss.str();
};
