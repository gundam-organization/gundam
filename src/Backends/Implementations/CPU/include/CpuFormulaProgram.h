#ifndef GUNDAM_CPU_FORMULA_PROGRAM_H
#define GUNDAM_CPU_FORMULA_PROGRAM_H

#include "EngineView.h"
#include "DialProgramValidation.h"
#include "Semantics/BackendDialSemantics.h"

#include <cmath>
#include <stdexcept>

namespace Backends {
  // No allocation, shared mutable state, or ROOT call in the interpreter.
  // The caller validates the program and owns stackSize doubles of scratch.
  template<typename InputReader>
  double evaluateFormula(const FormulaProgram& program_, InputReader readInput_, double* stack_) {
    std::size_t top{0};
    for( const auto& instruction : program_.instructions ){
      const auto op = instruction.opcode;
      if( op == FormulaOpcode::LoadInput ){ stack_[top++] = readInput_(instruction.operand); continue; }
      if( op == FormulaOpcode::LoadConstant ){ stack_[top++] = program_.constants[instruction.operand]; continue; }
      double& value = stack_[top - 1];
      switch( op ){
        case FormulaOpcode::Negate: value = -value; break;
        case FormulaOpcode::Exp: value = std::exp(value); break;
        case FormulaOpcode::Log: value = std::log(value); break;
        case FormulaOpcode::Log10: value = std::log10(value); break;
        case FormulaOpcode::Sqrt: value = std::sqrt(value); break;
        case FormulaOpcode::Abs: value = std::abs(value); break;
        case FormulaOpcode::Sin: value = std::sin(value); break;
        case FormulaOpcode::Cos: value = std::cos(value); break;
        case FormulaOpcode::Tan: value = std::tan(value); break;
        case FormulaOpcode::Asin: value = std::asin(value); break;
        case FormulaOpcode::Acos: value = std::acos(value); break;
        case FormulaOpcode::Atan: value = std::atan(value); break;
        case FormulaOpcode::Sinh: value = std::sinh(value); break;
        case FormulaOpcode::Cosh: value = std::cosh(value); break;
        case FormulaOpcode::Tanh: value = std::tanh(value); break;
        case FormulaOpcode::Erf: value = std::erf(value); break;
        case FormulaOpcode::Erfc: value = std::erfc(value); break;
        case FormulaOpcode::Floor: value = std::floor(value); break;
        case FormulaOpcode::Ceil: value = std::ceil(value); break;
        default: {
          const double right = value;
          double& left = stack_[top - 2];
          switch( op ){
            case FormulaOpcode::Add: left += right; break;
            case FormulaOpcode::Subtract: left -= right; break;
            case FormulaOpcode::Multiply: left *= right; break;
            case FormulaOpcode::Divide: left /= right; break;
            case FormulaOpcode::Pow: left = std::pow(left, right); break;
            case FormulaOpcode::Atan2: left = std::atan2(left, right); break;
            // Same comparison ordering as TMath::Min/Max, including NaN.
            case FormulaOpcode::Min: left = left <= right ? left : right; break;
            case FormulaOpcode::Max: left = left >= right ? left : right; break;
            default: throw std::runtime_error("Unknown formula bytecode opcode");
          }
          --top;
        }
      }
    }
    return stack_[0];
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
