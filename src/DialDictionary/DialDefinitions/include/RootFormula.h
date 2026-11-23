//
// Created by Adrien Blanchet on 10/04/2023.
//

/// @file RootFormula.h
/// @brief Declares RootFormula, a dial whose response is a ROOT TFormula of
///        the fit parameters.
/// @ingroup DialDictionary

#ifndef GUNDAM_ROOTFORMULA_H
#define GUNDAM_ROOTFORMULA_H

#include "DialBase.h"
#include "FormulaEvaluator.h"

#include <string>


/// Dial whose response is a ROOT TFormula of named inputs.
///
/// A dial is a response function of one or more fit parameters (see
/// DialBase). DialCollection::makeDial(const ConfigReader&, const Bin*)
/// creates a RootFormula for `dialType: Formula` (or `RootFormula`)
/// collections, with the expression from `options/formulaStr`. The formula
/// names its inputs as ROOT parameters in square brackets (for example
/// `[delta]`); positional variables such as `x[0]` are rejected. The input
/// names come from the `name` of each entry of the collection's
/// `dialInputList`, which is required, and the dial input buffer holds the
/// values in that order.
///
/// When the collection has a `binning`, DialCollection creates one dial per
/// bin. A formula parameter named after a bin variable is then a constant,
/// set to the bin centre (TFormula::SetParameter() through getFormula()).
///
/// Parsing, input binding and evaluation are done by a FormulaEvaluator
/// (src/Utils). Not supported by Cache::Manager (no GPU implementation).
/// @ingroup DialDictionary
class RootFormula : public DialBase {

public:
  /// Creates a dial with an empty formula. Call setFormulaStr() before
  /// evaluating.
  RootFormula() = default;

  /// Copies this formula dial.
  /// @return A new RootFormula with a copy of the formula, including its
  ///         parameter values and input binding.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<RootFormula>(*this); }

  /// Gives the type name of this dial.
  /// @return `"RootFormula"`.
  [[nodiscard]] std::string getDialTypeName() const override { return {"RootFormula"}; }

  /// Evaluates the formula at the current dial inputs.
  /// @param input_ Buffer holding one value per input name, in
  ///        `dialInputList` order.
  /// @return The formula value, with each named input taken from the buffer
  ///         and each constant (bin variable) kept at its stored value.
  /// @pre setInputNames() was called; otherwise, or if the buffer size
  ///      differs from the number of input names, it throws.
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  /// Describes the formula.
  /// @return `"RootFormula: formula: "` followed by the expanded expression
  ///         (TFormula::GetExpFormula()).
  [[nodiscard]] std::string getSummary() const override;

  /// Parses a formula expression and stores it in the dial.
  /// @param formulaStr_ ROOT TFormula expression, with its inputs written as
  ///        named parameters (`[name]`). It is also used as the TFormula
  ///        name, which is not added to ROOT's global list.
  /// @post Any previous formula, parameter values and input binding are
  ///       replaced; call setInputNames() again. Throws (LogThrowIf()) if
  ///       ROOT cannot parse the expression or if it uses positional
  ///       variables such as `x[0]`.
  void setFormulaStr(const std::string& formulaStr_);

  /// Binds the formula parameters to the dial inputs by name.
  /// @param names_ Names of the dial inputs, in the order of the input
  ///        buffer (the collection's `dialInputList`).
  /// @param constantNames_ Formula parameters that are not inputs and keep
  ///        their stored value: the bin variables of a binned collection.
  /// @pre setFormulaStr() was called.
  /// @post Throws if a name is empty or repeated, or if a formula parameter
  ///       is neither an input nor a constant.
  void setInputNames(const std::vector<std::string>& names_, const std::vector<std::string>& constantNames_ = {});

  /// Gives the number of inputs bound by setInputNames().
  /// @return The size of `names_` at the last setInputNames(); -1 before
  ///         then, or after setFormulaStr().
  [[nodiscard]] int getInputCount() const{ return _evaluator_.getInputCount(); }

  /// Gives mutable access to the formula, e.g. to set its parameters.
  /// DialCollection uses it to set the bin-centre parameters of binned
  /// formula dials.
  /// @return The TFormula held by the evaluator.
  TFormula& getFormula(){ return _evaluator_.getFormula(); }

  /// Gives read-only access to the formula. Used by getSummary().
  /// @return The TFormula held by the evaluator.
  const TFormula& getFormula() const{ return _evaluator_.getFormula(); }


private:
  /// Parses, binds and evaluates the formula (see FormulaEvaluator). Set by
  /// setFormulaStr() and setInputNames().
  FormulaEvaluator _evaluator_{};

};


#endif //GUNDAM_ROOTFORMULA_H
