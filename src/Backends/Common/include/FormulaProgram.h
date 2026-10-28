#ifndef GUNDAM_FORMULA_PROGRAM_H
#define GUNDAM_FORMULA_PROGRAM_H

#include "FormulaTypes.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace Backends {
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
