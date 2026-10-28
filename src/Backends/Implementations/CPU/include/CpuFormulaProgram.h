#ifndef GUNDAM_CPU_FORMULA_PROGRAM_H
#define GUNDAM_CPU_FORMULA_PROGRAM_H

#include "EngineView.h"
#include "DialProgramValidation.h"
#include "Semantics/BackendDialSemantics.h"
#include "Semantics/NativeFormulaProgram.h"

#include <cmath>
#include <stdexcept>

namespace Backends {
  // No allocation, shared mutable state, or ROOT call in the interpreter.
  // The caller validates the program and owns stackSize doubles of scratch.
  template<typename InputReader>
  double evaluateFormula(const FormulaProgram& program_, InputReader readInput_, double* stack_) {
    return evaluateNativeFormula(program_, readInput_, stack_, []() -> double {
          throw std::runtime_error("Unknown formula bytecode opcode");
        });
  }

  // Resolve the dial's bindings once, outside the opcode loop. All formulas
  // then use the same contiguous-input interpreter, including composite nodes.
  // Scratch contains stackSize + inputCount doubles and belongs to the worker.
  template<typename InputReader>
  double evaluateBoundFormula(const FormulaProgram& program_, InputReader readInput_, double* scratch_) {
    auto* values = scratch_ + program_.stackSize;
    for( std::size_t i = 0; i < program_.inputCount; ++i ){ values[i] = readInput_(i); }
    return evaluateFormula(program_, [values](std::size_t index_){ return values[index_]; }, scratch_);
  }

  inline double evaluateFormulaDial(const PropagationView& model_, const BackendDialDescriptor& dial_,
                                    const double* parameters_, double* stack_) {
    const auto& program = model_.formulaPrograms[dial_.formulaProgramIndex];
    const auto read = [&](std::size_t index_) {
      const auto& binding = model_.formulaInputs[dial_.firstFormulaInput + index_];
      if( binding.parameterIndex < 0 ){ return binding.constant; }
      const auto& input = model_.dialInputs[dial_.firstInput + binding.parameterIndex];
      return Semantics::transformDialInput(input, parameters_[input.parameterIndex]);
    };
    return Semantics::clampDialResponse(dial_, evaluateBoundFormula(program, read, stack_));
  }
}

#endif
