#ifndef GUNDAM_NATIVE_FORMULA_PLAN_H
#define GUNDAM_NATIVE_FORMULA_PLAN_H
#include "FormulaTypes.h"
#include <cstdint>
namespace Backends {
  enum class NativeFormulaKernel : std::uint32_t { Interpreter, Input, Constant, Binary, MultiplyThenAdd };
  struct NativeFormulaPlan {
    NativeFormulaKernel kernel{NativeFormulaKernel::Interpreter};
    FormulaOpcode operation{FormulaOpcode::Add};
    std::uint32_t inputs[3]{};
    std::uint32_t literalMask{0};
    double literals[3]{};
  };
}
#endif
