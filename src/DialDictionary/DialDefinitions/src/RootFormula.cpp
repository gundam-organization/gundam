//
// Created by Adrien Blanchet on 10/04/2023.
//

#include "RootFormula.h"

#include <sstream>

bool RootFormula::hasAnalyticGradient() const {
  _evaluator_.prepareGradient();
  return true;
}

double RootFormula::evalGradient(const DialInputBuffer& input_, int iInput_) const {
  return _evaluator_.evalGradient(input_.getInputBuffer(), iInput_);
}

double RootFormula::evalResponse(const DialInputBuffer& input_) const {
  return _evaluator_.eval(input_.getInputBuffer());
}

void RootFormula::setFormulaStr(const std::string& formulaStr_){
  _evaluator_.setFormulaStr(formulaStr_);
}

void RootFormula::setInputNames(const std::vector<std::string>& names_, const std::vector<std::string>& constantNames_){
  _evaluator_.setInputNames(names_, constantNames_);
}

std::string RootFormula::getSummary() const {
  std::stringstream ss;
  ss << this->getDialTypeName() << ": formula: " << getFormula().GetExpFormula();
  return ss.str();
}
