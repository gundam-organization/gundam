#ifndef GUNDAM_FORMULA_TYPES_H
#define GUNDAM_FORMULA_TYPES_H
#ifndef __METAL_VERSION__
#include <cstdint>
#endif
namespace Backends {
#ifdef __METAL_VERSION__
  using FormulaIndex = uint;
#else
  using FormulaIndex = std::uint32_t;
#endif
  // Fixed-width instructions contain no host pointers. Only the owning
  // containers need packing when transferring a program to a device.
  enum class FormulaOpcode : FormulaIndex {
    LoadInput, LoadConstant, Add, Subtract, Multiply, Divide, Negate, Pow,
    Exp, Log, Log10, Sqrt, Abs, Sin, Cos, Tan, Asin, Acos, Atan, Atan2,
    Sinh, Cosh, Tanh, Erf, Erfc, Floor, Ceil, Min, Max
  };

  struct FormulaInstruction {
    FormulaOpcode opcode{FormulaOpcode::LoadConstant};
    FormulaIndex operand{0};
  };

}
#endif
