#ifndef GUNDAM_FORMULA_PROGRAM_H
#define GUNDAM_FORMULA_PROGRAM_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace Backends {
  // Fixed-width instructions contain no host pointers. Only the owning
  // containers need packing when transferring a program to a device.
  enum class FormulaOpcode : std::uint32_t {
    LoadInput, LoadConstant, Add, Subtract, Multiply, Divide, Negate, Pow,
    Exp, Log, Log10, Sqrt, Abs, Sin, Cos, Tan, Asin, Acos, Atan, Atan2,
    Sinh, Cosh, Tanh, Erf, Erfc, Floor, Ceil, Min, Max
  };

  struct FormulaInstruction {
    FormulaOpcode opcode{FormulaOpcode::LoadConstant};
    std::uint32_t operand{0};
  };

  struct FormulaProgram {
    std::vector<FormulaInstruction> instructions;
    std::vector<double> constants;
    std::size_t inputCount{0};
    std::size_t stackSize{0};

    // Validate untrusted descriptors once, before entering the hot loop.
    void validate() const;
  };

  struct FormulaInput {
    // Relative index in the dial's ordinary parameter inputs; -1 denotes a
    // loaded event value, already mirrored by EventFormula during binding.
    int parameterIndex{-1};
    double constant{0};
  };
}

#endif
