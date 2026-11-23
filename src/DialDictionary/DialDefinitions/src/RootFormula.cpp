//
// Created by Adrien Blanchet on 10/04/2023.
//

/// @file RootFormula.cpp
/// @brief Implements RootFormula: formula parsing, evaluation and summary.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in RootFormula.h. The
/// comments here describe how each method does its job.

#include "RootFormula.h"

#include <sstream>

/// @par Implementation
/// Delegates to FormulaEvaluator::eval(). When the inputs are in the same
/// order as the formula parameters, it passes the buffer as the parameter
/// array directly; otherwise it fills a parameter array, taking each value
/// from the buffer or, for a constant, from the stored parameter.
double RootFormula::evalResponse(const DialInputBuffer& input_) const {
  return _evaluator_.eval(input_.getInputBuffer());
}


/// @par Implementation
/// Delegates to FormulaEvaluator::setFormulaStr(), which builds the TFormula
/// without registering it in ROOT's global list and clears the input binding.
void RootFormula::setFormulaStr(const std::string& formulaStr_){
  _evaluator_.setFormulaStr(formulaStr_);
}

/// @par Implementation
/// Delegates to FormulaEvaluator::setInputNames().
void RootFormula::setInputNames(const std::vector<std::string>& names_, const std::vector<std::string>& constantNames_){
  _evaluator_.setInputNames(names_, constantNames_);
}

/// @par Implementation
/// Streams getDialTypeName() and TFormula::GetExpFormula() into a string.
std::string RootFormula::getSummary() const {
  std::stringstream ss;
  ss << this->getDialTypeName() << ": formula: " << getFormula().GetExpFormula();
  return ss.str();
}
