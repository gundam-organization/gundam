#ifndef GUNDAM_FORMULA_COMPILER_H
#define GUNDAM_FORMULA_COMPILER_H

#include "FormulaProgram.h"

#include <map>
#include <string>
#include <vector>

namespace Backends {
  // Pure arithmetic subset of TFormula: named inputs, numeric literals,
  // + - * /, right-associative ^ / **, and an explicit math function list.
  // Unsupported ROOT/C++ constructs are errors, never a fallback to ROOT.
  class FormulaCompiler {
  public:
    static FormulaProgram compile(const std::string& expression_,
                                  const std::vector<std::string>& inputNames_,
                                  const std::map<std::string, double>& constants_ = {});
  };
}

#endif
