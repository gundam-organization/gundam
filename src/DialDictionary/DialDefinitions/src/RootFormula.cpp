//
// Created by Adrien Blanchet on 10/04/2023.
//

#include "RootFormula.h"

#include <algorithm>
#include <map>

double RootFormula::evalResponse(const DialInputBuffer& input_) const {
  LogThrowIf(_inputCount_ < 0 or input_.getInputBuffer().size() != size_t(_inputCount_),
             "Formula: named input count mismatch");
  if( _isInputOrderMatching_ ){ return _formula_.EvalPar(nullptr, input_.getInputBuffer().data()); }
  std::vector<double> parameters(_formula_.GetNpar());
  for( size_t i = 0 ; i < parameters.size() ; ++i ){
    const int inputIndex = _parameterInputIndices_[i];
    parameters[i] = inputIndex >= 0 ? input_.getInputBuffer()[inputIndex] : _formula_.GetParameter(int(i));
  }
  return _formula_.EvalPar(nullptr, parameters.data());
}


void RootFormula::setFormulaStr(const std::string& formulaStr_){
  // Dial formulas are independent. Registering an expression such as x[0]
  // as a global formula name can make later nodes recursively expand it.
  _formula_ = TFormula(formulaStr_.c_str(), formulaStr_.c_str(), false);
  LogThrowIf(not _formula_.IsValid(), "\"" << formulaStr_ << "\": could not be parsed as formula expression.");
  LogThrowIf(_formula_.GetNdim() != 0,
             "Formula dials require named inputs such as [delta]; positional x[i] inputs are unsupported");
  _inputCount_ = -1;
  _isInputOrderMatching_ = false;
  _parameterInputIndices_.clear();
}

void RootFormula::setInputNames(const std::vector<std::string>& names_, const std::vector<std::string>& constantNames_){
  std::map<std::string, int> inputIndices;
  for( size_t i = 0 ; i < names_.size() ; ++i ){
    LogThrowIf(names_[i].empty(), "Formula input name must not be empty");
    LogThrowIf(not inputIndices.emplace(names_[i], int(i)).second, "Formula: duplicate input name: " << names_[i]);
  }
  _inputCount_ = int(names_.size());
  _parameterInputIndices_.assign(_formula_.GetNpar(), -1);
  for( int i = 0 ; i < _formula_.GetNpar() ; ++i ){
    const std::string name = _formula_.GetParName(i);
    const auto input = inputIndices.find(name);
    if( input != inputIndices.end() ){ _parameterInputIndices_[i] = input->second; }
    else{
      LogThrowIf(std::find(constantNames_.begin(), constantNames_.end(), name) == constantNames_.end(),
                 "Formula: undeclared input [" << name << "]");
    }
  }
  _isInputOrderMatching_ = _formula_.GetNpar() == _inputCount_;
  for( int i = 0 ; i < _formula_.GetNpar() ; ++i ){
    _isInputOrderMatching_ = _isInputOrderMatching_ and _parameterInputIndices_[i] == i;
  }
}

std::string RootFormula::getSummary() const {
  std::stringstream ss;
  ss << this->getDialTypeName() << ": formula: " << _formula_.GetExpFormula();
  return ss.str();
};
