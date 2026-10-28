#ifndef GUNDAM_NATIVE_COMPOSITE_PROGRAM_H
#define GUNDAM_NATIVE_COMPOSITE_PROGRAM_H

#include "Semantics/BackendDialSemanticsCore.h"

namespace Backends {
  // Shared CPU/CUDA graph traversal. All adapters are compile-time types; input
  // storage, formula binding strategy and surface kernels remain backend-owned.
  // Operations is a stateless policy: it adds neither arguments nor indirect calls.
  template<typename Operations, typename Model>
  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double evaluateNativeComposite(const Model& model_, const BackendDialDescriptor& root_,
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
      const auto* data = dial.payloadSize ? Operations::payloads(model_) + dial.payloadOffset : nullptr;
      double response;
      if( dial.type == BackendDialType::Shift ){ response = data[0]; }
      else if( dial.type == BackendDialType::Formula ){
        const auto read = [&](std::size_t index_) {
          const auto& binding = model_.formulaInputs[dial.firstFormulaInput + index_];
          return binding.parameterIndex < 0 ? binding.constant : load(binding.parameterIndex);
        };
        response = Operations::formula(model_, dial, read, formulaScratch_);
      }
      else{
        double x = load(0);
        if( dial.type == BackendDialType::Norm ){ response = x; }
        else if( dial.type == BackendDialType::Bilinear or dial.type == BackendDialType::Bicubic ){
          double y = load(1);
          response = Operations::surface(dial, x, y, data);
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
          if constexpr( Operations::checkSplineInput ){
            if( dial.type == BackendDialType::CompactSpline or dial.type == BackendDialType::MonotonicSpline
                or dial.type == BackendDialType::UniformSpline ){
              const double position = (x - data[0]) / data[1];
              const double limit = dial.type == BackendDialType::UniformSpline ? 1073741800. : 2147483000.;
              if( not (position >= -limit and position <= limit) ){ return NAN; }
            }
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
            default: return Operations::invalidOpcode();
          }
        }
      }
      scratch_[program.inputCount + i] = Semantics::clampDialResponse(dial, response);
    }
    return Semantics::clampDialResponse(root_, scratch_[program.outputSlot]);
  }
}
#endif
