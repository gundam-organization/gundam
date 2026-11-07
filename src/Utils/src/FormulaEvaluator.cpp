#include "FormulaEvaluator.h"

#include "Logger.h"

#include <algorithm>
#include <map>
#include <utility>

void FormulaEvaluator::setFormulaStr(const std::string& formulaStr_){
  _inputCount_ = -1;
  _isInputOrderMatching_ = false;
  _parameterInputIndices_.clear();
  // Expressions must not register themselves by name in ROOT's global
  // registry, including when loaders compile the same text independently.
  _formula_ = TFormula(formulaStr_.c_str(), formulaStr_.c_str(), false);
  LogThrowIf(not _formula_.IsValid(), "\"" << formulaStr_ << "\": could not be parsed as formula expression.");
  LogThrowIf(_formula_.GetNdim() != 0,
             "Formula evaluation requires named inputs such as [delta]; positional x[i] inputs are unsupported");
}

void FormulaEvaluator::setInputNames(const std::vector<std::string>& names_, const std::vector<std::string>& constantNames_){
  _inputCount_ = -1;
  _isInputOrderMatching_ = false;
  _parameterInputIndices_.clear();
  LogThrowIf(not _formula_.IsValid(), "Formula: compile an expression before binding inputs");
  std::map<std::string, int> inputIndices;
  for( size_t i = 0 ; i < names_.size() ; ++i ){
    LogThrowIf(names_[i].empty(), "Formula input name must not be empty");
    LogThrowIf(not inputIndices.emplace(names_[i], int(i)).second, "Formula: duplicate input name: " << names_[i]);
  }
  std::vector<int> parameterInputIndices(_formula_.GetNpar(), -1);
  for( int i = 0 ; i < _formula_.GetNpar() ; ++i ){
    const std::string name = _formula_.GetParName(i);
    const auto input = inputIndices.find(name);
    if( input != inputIndices.end() ){ parameterInputIndices[i] = input->second; }
    else{
      LogThrowIf(std::find(constantNames_.begin(), constantNames_.end(), name) == constantNames_.end(),
                 "Formula: undeclared input [" << name << "]");
    }
  }
  _inputCount_ = int(names_.size());
  _parameterInputIndices_ = std::move(parameterInputIndices);
  _isInputOrderMatching_ = _formula_.GetNpar() == _inputCount_;
  for( int i = 0 ; i < _formula_.GetNpar() ; ++i ){
    _isInputOrderMatching_ = _isInputOrderMatching_ and _parameterInputIndices_[i] == i;
  }
}

void FormulaEvaluator::checkInputsBound() const {
  LogThrowIf(_inputCount_ < 0, "Formula: input names have not been bound");
}

double FormulaEvaluator::eval(const std::vector<double>& inputs_) const {
  checkInputsBound();
  LogThrowIf(inputs_.size() != size_t(_inputCount_), "Formula: named input count mismatch");
  if( _isInputOrderMatching_ ){ return _formula_.EvalPar(nullptr, inputs_.data()); }
  return evalWithInputs([&](int index_){ return inputs_[index_]; });
}
