#ifndef GUNDAM_NATIVE_FORMULA_PROGRAM_H
#define GUNDAM_NATIVE_FORMULA_PROGRAM_H

#include "FormulaTypes.h"
#include "Semantics/BackendSemanticsQualifiers.h"

#include <cmath>
#include <cstddef>

namespace Backends::NativeFormulaMath {
  // Statically selected native-double operations; no virtual/indirect dispatch.
#define GUNDAM_NATIVE_FORMULA_UNARY(name) \
  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE \
  double name(double value_) { return ::name(value_); }
  GUNDAM_NATIVE_FORMULA_UNARY(exp)
  GUNDAM_NATIVE_FORMULA_UNARY(log)
  GUNDAM_NATIVE_FORMULA_UNARY(log10)
  GUNDAM_NATIVE_FORMULA_UNARY(sqrt)
  GUNDAM_NATIVE_FORMULA_UNARY(sin)
  GUNDAM_NATIVE_FORMULA_UNARY(cos)
  GUNDAM_NATIVE_FORMULA_UNARY(tan)
  GUNDAM_NATIVE_FORMULA_UNARY(asin)
  GUNDAM_NATIVE_FORMULA_UNARY(acos)
  GUNDAM_NATIVE_FORMULA_UNARY(atan)
  GUNDAM_NATIVE_FORMULA_UNARY(sinh)
  GUNDAM_NATIVE_FORMULA_UNARY(cosh)
  GUNDAM_NATIVE_FORMULA_UNARY(tanh)
  GUNDAM_NATIVE_FORMULA_UNARY(erf)
  GUNDAM_NATIVE_FORMULA_UNARY(erfc)
  GUNDAM_NATIVE_FORMULA_UNARY(floor)
  GUNDAM_NATIVE_FORMULA_UNARY(ceil)
#undef GUNDAM_NATIVE_FORMULA_UNARY
  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double negate(double value_) { return -value_; }
  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double abs(double value_) { return ::fabs(value_); }
  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double add(double left_, double right_) { return left_ + right_; }
  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double subtract(double left_, double right_) { return left_ - right_; }
  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double multiply(double left_, double right_) { return left_ * right_; }
  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double divide(double left_, double right_) { return left_ / right_; }
  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double pow(double left_, double right_) { return ::pow(left_, right_); }
  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double atan2(double left_, double right_) { return ::atan2(left_, right_); }
  // Preserve TMath comparison ordering, including NaN and signed zero.
  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double min(double left_, double right_) { return left_ <= right_ ? left_ : right_; }
  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double max(double left_, double right_) { return left_ >= right_ ? left_ : right_; }
}

namespace Backends {
  // A range view lets CUDA use packed tables without changing CPU's existing
  // FormulaProgram calling convention or introducing temporary input copies.
  struct NativeFormulaInstructions {
    const FormulaInstruction* first;
    const FormulaInstruction* last;
    GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
    const FormulaInstruction* begin() const { return first; }
    GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
    const FormulaInstruction* end() const { return last; }
  };
  struct NativeFormulaView {
    NativeFormulaInstructions instructions;
    const double* constants;
  };

  // Callers retain their input binding strategy and own the stack. The same
  // loop accepts CPU containers or non-owning CUDA ranges without allocating.
  template<typename Program, typename InputReader, typename InvalidOpcode>
  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double evaluateNativeFormula(const Program& program_, InputReader readInput_,
      double* stack_, InvalidOpcode invalidOpcode_) {
    std::size_t top{0};
    for( const auto& instruction : program_.instructions ){
      const auto op = instruction.opcode;
      if( op == FormulaOpcode::LoadInput ){ stack_[top++] = readInput_(instruction.operand); continue; }
      if( op == FormulaOpcode::LoadConstant ){ stack_[top++] = program_.constants[instruction.operand]; continue; }
      double& value = stack_[top - 1];
      switch( op ){
#define GUNDAM_FORMULA_UNARY(opcode, operation) \
        case FormulaOpcode::opcode: value = NativeFormulaMath::operation(value); break;
#include "DeviceMath/SharedFormulaOperations.h"
#undef GUNDAM_FORMULA_UNARY
        default: {
          const double right = value;
          double& left = stack_[top - 2];
          switch( op ){
#define GUNDAM_FORMULA_BINARY(opcode, operation) \
            case FormulaOpcode::opcode: left = NativeFormulaMath::operation(left, right); break;
#include "DeviceMath/SharedFormulaOperations.h"
#undef GUNDAM_FORMULA_BINARY
            default: return invalidOpcode_();
          }
          --top;
        }
      }
    }
    return stack_[0];
  }
}
#endif
