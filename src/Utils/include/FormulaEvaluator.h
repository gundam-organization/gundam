#ifndef GUNDAM_FORMULA_EVALUATOR_H
#define GUNDAM_FORMULA_EVALUATOR_H

#include <TFormula.h>

#include <string>
#include <vector>

// Compiles named ROOT parameters and binds them to a caller-owned input
// layout. Evaluation is read-only; constants are set only during preparation.
// This utility does not depend on events, fit parameters or dials.
class FormulaEvaluator {
public:
  void setFormulaStr(const std::string& formulaStr_);
  void setInputNames(const std::vector<std::string>& names_, const std::vector<std::string>& constantNames_ = {});

  [[nodiscard]] int getInputCount() const{ return _inputCount_; }
  TFormula& getFormula(){ return _formula_; }
  const TFormula& getFormula() const{ return _formula_; }

  [[nodiscard]] double eval(const std::vector<double>& inputs_) const;

  // Read only inputs referenced by the expression, without copying the
  // complete source layout (e.g. an event's variable collection).
  template<typename InputReader>
  [[nodiscard]] double evalWithInputs(const InputReader& readInput_) const {
    checkInputsBound();
    std::vector<double> parameters(_parameterInputIndices_.size());
    for( size_t i = 0 ; i < parameters.size() ; ++i ){
      const int index = _parameterInputIndices_[i];
      parameters[i] = index >= 0 ? readInput_(index) : _formula_.GetParameter(int(i));
    }
    return _formula_.EvalPar(nullptr, parameters.data());
  }

private:
  void checkInputsBound() const;

  TFormula _formula_{};
  int _inputCount_{-1};
  bool _isInputOrderMatching_{false};
  // One entry per ROOT parameter; -1 keeps a configured constant.
  std::vector<int> _parameterInputIndices_;
};

#endif
