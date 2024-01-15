//
// Created by Nadrino on 10/04/2023.
//

#include "RootFormula.h"


double RootFormula::evalResponse(const DialInputBuffer& input_) const {
  return _formula_.EvalPar(input_.getBuffer());
}


void RootFormula::setFormulaStr(const std::string& formulaStr_){
  _formula_ = TFormula(formulaStr_.c_str(), formulaStr_.c_str());
  LogThrowIf(not _formula_.IsValid(), "\"" << formulaStr_ << "\": could not be parsed as formula expression.");
}
