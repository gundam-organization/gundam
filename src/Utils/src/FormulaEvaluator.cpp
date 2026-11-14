#include "FormulaEvaluator.h"

#include "Logger.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <utility>

struct FormulaEvaluator::GradientState {
  std::mutex mutex;
  std::unique_ptr<TFormula> formula;
};

FormulaEvaluator::FormulaEvaluator() : _gradientState_(std::make_shared<GradientState>()) {}

void FormulaEvaluator::setFormulaStr(const std::string& formulaStr_){
  _gradientState_ = std::make_shared<GradientState>();
  _inputCount_ = -1;
  _isInputOrderMatching_ = false;
  _parameterInputIndices_.clear();
  // Expressions must not register themselves by name in ROOT's global
  // registry, including when loaders compile the same text independently.
  _formula_ = TFormula(formulaStr_.c_str(), formulaStr_.c_str(), false);
  _formulaStr_ = formulaStr_;
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

void FormulaEvaluator::prepareGradient() const {
  checkInputsBound();
  std::lock_guard<std::mutex> lock(_gradientState_->mutex);
  if( _gradientState_->formula ){ return; }
  // ROOT's interpreter/Clad generation must not run concurrently, including
  // when independent evaluators are being prepared by different callers.
  static std::mutex generationMutex;
  std::lock_guard<std::mutex> generationLock(generationMutex);
  auto formula = std::make_unique<TFormula>(_formula_);
  if( formula->GetNpar() != 0 ){
    LogThrowIf(not formula->GenerateGradientPar() or not formula->HasGeneratedGradient(),
               "Cannot generate analytic gradient for formula: " << _formulaStr_);
    // Clad can emit numerical differentiation for unsupported external calls.
    // Such generated routines do not satisfy the analytic-gradient contract.
    const auto source = formula->GetGradientFormula();
    LogThrowIf(source.Contains("numerical_diff"),
               "ROOT generated a numerical fallback for formula: " << _formulaStr_);
  }
  _gradientState_->formula = std::move(formula);
}

double FormulaEvaluator::evalGradient(const std::vector<double>& inputs_, int iInput_) const {
  checkInputsBound();
  LogThrowIf(inputs_.size() != size_t(_inputCount_), "Formula: named input count mismatch");
  std::lock_guard<std::mutex> lock(_gradientState_->mutex);
  LogThrowIf(not _gradientState_->formula, "Formula gradient has not been prepared: " << _formulaStr_);
  if( iInput_ < 0 or iInput_ >= _inputCount_ ){ return 0.; }
  const auto entry = std::find(_parameterInputIndices_.begin(), _parameterInputIndices_.end(), iInput_);
  if( entry == _parameterInputIndices_.end() ){ return 0.; }
  std::vector<double> parameters(_parameterInputIndices_.size());
  for( size_t i = 0; i < parameters.size(); ++i ){
    const int index = _parameterInputIndices_[i];
    parameters[i] = index >= 0 ? inputs_[index] : _formula_.GetParameter(int(i));
  }
  // GradientPar has no explicit parameter-array argument. Only this dedicated
  // copy is modified, under the lock shared by all evaluator clones.
  auto& formula = *_gradientState_->formula;
  formula.SetParameters(parameters.data());
  std::vector<double> gradient(parameters.size(), 0.);
  formula.GradientPar(nullptr, gradient.data());
  return gradient[size_t(entry - _parameterInputIndices_.begin())];
}
