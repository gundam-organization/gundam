#ifndef GUNDAM_CUDA_DIAL_PROGRAMS_H
#define GUNDAM_CUDA_DIAL_PROGRAMS_H

#include "CudaWeightModel.h"
#include "Semantics/BackendDialSemanticsCore.h"

#include <cmath>

namespace Backends {
  // Non-owning, trivially copyable kernel argument; every pointer addresses device memory.
  struct CudaProgramView {
    const CudaFormulaProgram* formulaPrograms{};
    const FormulaInstruction* formulaInstructions{};
    const double* formulaConstants{};
    const FormulaInput* formulaInputs{};
    const CompositeProgram* compositePrograms{};
    const CompositeInstruction* compositeInstructions{};
    const BackendDialInputDescriptor* compositeInputs{};
    const BackendDialInputDescriptor* dialInputs{};
    const double* payloads{};
    const std::size_t* scratchOffsets{};
    double* scratch{};
  };

  template<typename InputReader>
  GUNDAM_BACKEND_HOST_DEVICE inline double evaluateCudaFormula(
      const CudaFormulaProgram& program_, const FormulaInstruction* instructions_,
      const double* constants_, InputReader readInput_, double* stack_) {
    std::size_t top{0};
    for( std::size_t i = 0; i < program_.instructionCount; ++i ){
      const auto& instruction = instructions_[program_.firstInstruction + i];
      const auto op = instruction.opcode;
      if( op == FormulaOpcode::LoadInput ){ stack_[top++] = readInput_(instruction.operand); continue; }
      if( op == FormulaOpcode::LoadConstant ){ stack_[top++] = constants_[program_.firstConstant + instruction.operand]; continue; }
      double& value = stack_[top - 1];
      switch( op ){
        case FormulaOpcode::Negate: value = -value; break;
        case FormulaOpcode::Exp: value = ::exp(value); break;
        case FormulaOpcode::Log: value = ::log(value); break;
        case FormulaOpcode::Log10: value = ::log10(value); break;
        case FormulaOpcode::Sqrt: value = ::sqrt(value); break;
        case FormulaOpcode::Abs: value = ::fabs(value); break;
        case FormulaOpcode::Sin: value = ::sin(value); break;
        case FormulaOpcode::Cos: value = ::cos(value); break;
        case FormulaOpcode::Tan: value = ::tan(value); break;
        case FormulaOpcode::Asin: value = ::asin(value); break;
        case FormulaOpcode::Acos: value = ::acos(value); break;
        case FormulaOpcode::Atan: value = ::atan(value); break;
        case FormulaOpcode::Sinh: value = ::sinh(value); break;
        case FormulaOpcode::Cosh: value = ::cosh(value); break;
        case FormulaOpcode::Tanh: value = ::tanh(value); break;
        case FormulaOpcode::Erf: value = ::erf(value); break;
        case FormulaOpcode::Erfc: value = ::erfc(value); break;
        case FormulaOpcode::Floor: value = ::floor(value); break;
        case FormulaOpcode::Ceil: value = ::ceil(value); break;
        default: {
          const double right = value;
          double& left = stack_[top - 2];
          switch( op ){
            case FormulaOpcode::Add: left += right; break;
            case FormulaOpcode::Subtract: left -= right; break;
            case FormulaOpcode::Multiply: left *= right; break;
            case FormulaOpcode::Divide: left /= right; break;
            case FormulaOpcode::Pow: left = ::pow(left, right); break;
            case FormulaOpcode::Atan2: left = ::atan2(left, right); break;
            // Same comparison ordering as TMath::Min/Max, including NaN.
            case FormulaOpcode::Min: left = left <= right ? left : right; break;
            case FormulaOpcode::Max: left = left >= right ? left : right; break;
            default: GUNDAM_BACKEND_SEMANTICS_ASSERT(false); return NAN;
          }
          --top;
        }
      }
    }
    return stack_[0];
  }

  GUNDAM_BACKEND_HOST_DEVICE inline double evaluateCudaComposite(const CudaProgramView& model_, const BackendDialDescriptor& root_,
                                  const double* parameters_, double* scratch_, double* formulaScratch_) {
    const auto& program = model_.compositePrograms[root_.compositeProgram];
    for( std::size_t i = 0; i < program.inputCount; ++i ){
      const auto& input = model_.dialInputs[root_.firstInput + i];
      scratch_[i] = Semantics::transformDialInput(input, parameters_[input.parameterIndex]);
    }
    for( std::size_t i = 0; i < program.instructionCount; ++i ){
      const auto& instruction = model_.compositeInstructions[program.firstInstruction + i];
      const auto& dial = instruction.dial;
      auto load = [&](std::size_t index_) {
        const auto& input = model_.compositeInputs[dial.firstInput + index_];
        double value = scratch_[input.parameterIndex];
        if( input.useMirror ){
          // Preserve CompositeDial's operations, including nested boundaries.
          value = ::fabs(::fmod(value - input.mirrorMin, 2 * input.mirrorRange));
          if( value > input.mirrorRange ){ value = 2 * input.mirrorRange - value; }
          value += input.mirrorMin;
        }
        return value;
      };
      const auto* data = dial.payloadSize ? model_.payloads + dial.payloadOffset : nullptr;
      double response;
      if( dial.type == BackendDialType::Shift ){ response = data[0]; }
      else if( dial.type == BackendDialType::Formula ){
        const auto read = [&](std::size_t index_) {
          const auto& binding = model_.formulaInputs[dial.firstFormulaInput + index_];
          return binding.parameterIndex < 0 ? binding.constant : load(binding.parameterIndex);
        };
        response = evaluateCudaFormula(model_.formulaPrograms[dial.formulaProgramIndex],
            model_.formulaInstructions, model_.formulaConstants, read, formulaScratch_);
      }
      else{
        double x = load(0);
        if( dial.type == BackendDialType::Norm ){ response = x; }
        else if( dial.type == BackendDialType::Bilinear or dial.type == BackendDialType::Bicubic ){
          double y = load(1);
          response = GundamDeviceMath::EvaluateSurface(x, y, data,
              dial.type == BackendDialType::Bicubic, dial.allowExtrapolation);
        }
        else if( dial.type == BackendDialType::Graph ){
          // Graph returns endpoint ordinates directly when extrapolation is off.
          if( not dial.allowExtrapolation and x <= data[1] ){ response = data[0]; }
          else if( not dial.allowExtrapolation and x >= data[dial.payloadSize - 1] ){ response = data[dial.payloadSize - 2]; }
          else{ response = CalculateGraph(x, -1E20, 1E20, data, int(dial.payloadSize)); }
        }
        else{
          if( not dial.allowExtrapolation ){
            if( x <= instruction.inputMin ){ x = instruction.inputMin; }
            else if( x >= instruction.inputMax ){ x = instruction.inputMax; }
          }
          if( dial.type == BackendDialType::CompactSpline or dial.type == BackendDialType::MonotonicSpline
              or dial.type == BackendDialType::UniformSpline ){
            const double position = (x - data[0]) / data[1];
            const double limit = dial.type == BackendDialType::UniformSpline ? 1073741800. : 2147483000.;
            // Avoid an invalid floating-to-int conversion in the spline evaluator.
            if( not (position >= -limit and position <= limit) ){ return NAN; }
          }
          switch( dial.type ){
            case BackendDialType::CompactSpline:
              response = CalculateCompactSpline(x, -1E20, 1E20, data, int(dial.payloadSize - 2)); break;
            case BackendDialType::UniformSpline:
              response = CalculateUniformSpline(x, -1E20, 1E20, data, int(dial.payloadSize)); break;
            case BackendDialType::MonotonicSpline:
              response = CalculateMonotonicSpline(x, -1E20, 1E20, data, int(dial.payloadSize - 2)); break;
            case BackendDialType::GeneralSpline:
              response = CalculateGeneralSpline(x, -1E20, 1E20, data, int(dial.payloadSize)); break;
            default: GUNDAM_BACKEND_SEMANTICS_ASSERT(false); return NAN;
          }
        }
      }
      scratch_[program.inputCount + i] = Semantics::clampDialResponse(dial, response);
    }
    return Semantics::clampDialResponse(root_, scratch_[program.outputSlot]);
  }

  GUNDAM_BACKEND_HOST_DEVICE inline double evaluateCudaProgram(
      const CudaProgramView& view_, const BackendDialDescriptor& dial_,
      std::size_t index_, const double* parameters_) {
    auto* scratch = view_.scratch + view_.scratchOffsets[index_];
    if( dial_.type == BackendDialType::Composite ){
      const auto& program = view_.compositePrograms[dial_.compositeProgram];
      return evaluateCudaComposite(view_, dial_, parameters_, scratch,
                                   scratch + program.inputCount + program.instructionCount);
    }
    for( std::size_t i = 0; i < dial_.inputCount; ++i ){
      const auto& input = view_.dialInputs[dial_.firstInput + i];
      scratch[i] = Semantics::transformDialInput(input, parameters_[input.parameterIndex]);
    }
    const auto read = [&](std::size_t i_) {
      const auto& binding = view_.formulaInputs[dial_.firstFormulaInput + i_];
      return binding.parameterIndex < 0 ? binding.constant : scratch[binding.parameterIndex];
    };
    return Semantics::clampDialResponse(dial_, evaluateCudaFormula(
        view_.formulaPrograms[dial_.formulaProgramIndex], view_.formulaInstructions,
        view_.formulaConstants, read, scratch + dial_.inputCount));
  }
}
#endif
