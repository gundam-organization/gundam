#ifndef GUNDAM_CPU_COMPOSITE_PROGRAM_H
#define GUNDAM_CPU_COMPOSITE_PROGRAM_H

#include "EngineView.h"
#include "CpuFormulaProgram.h"
#include "Semantics/BackendDialSemantics.h"
#include "CalculateBilinearInterpolation.h"
#include "CalculateBicubicSpline.h"

#include <cmath>

namespace Backends {
  inline void validateCompositePrograms(const PropagationView& model_) {
    for( const auto& program : model_.compositePrograms ){
      if( program.firstInstruction > model_.compositeInstructions.size()
          or program.instructionCount > model_.compositeInstructions.size() - program.firstInstruction
          or program.instructionCount > std::size_t(-1) - program.inputCount
          or program.outputSlot >= program.scratchSize() ){
        throw std::runtime_error("Invalid CPU composite program range.");
      }
      for( std::size_t i = 0; i < program.instructionCount; ++i ){
        const auto& dial = model_.compositeInstructions[program.firstInstruction + i].dial;
        const bool surface = dial.type == BackendDialType::Bilinear or dial.type == BackendDialType::Bicubic;
        std::size_t minimumPayload{0};
        switch( dial.type ){
          case BackendDialType::Norm: break;
          case BackendDialType::Formula: break;
          case BackendDialType::Shift: minimumPayload = 1; break;
          case BackendDialType::Graph: minimumPayload = 2; break;
          case BackendDialType::CompactSpline:
          case BackendDialType::MonotonicSpline: minimumPayload = 3; break;
          case BackendDialType::UniformSpline: minimumPayload = 4; break;
          case BackendDialType::GeneralSpline: minimumPayload = 5; break;
          case BackendDialType::Bilinear:
          case BackendDialType::Bicubic: minimumPayload = 10; break;
          default: throw std::runtime_error("Unsupported CPU composite instruction type.");
        }
        if( dial.firstInput > model_.compositeInputs.size()
            or dial.inputCount > model_.compositeInputs.size() - dial.firstInput
            or (dial.type != BackendDialType::Shift and dial.type != BackendDialType::Formula
                and dial.inputCount != (surface ? 2u : 1u))
            or dial.payloadOffset > model_.dialPayloads.size()
            or dial.payloadSize > model_.dialPayloads.size() - dial.payloadOffset
            or dial.payloadSize < minimumPayload ){
          throw std::runtime_error("Invalid CPU composite instruction layout.");
        }
        for( std::size_t j = 0; j < dial.inputCount; ++j ){
          const auto& input = model_.compositeInputs[dial.firstInput + j];
          if( input.source != DialInputSource::Parameter or input.parameterIndex >= program.inputCount + i ){
            throw std::runtime_error("CPU composite inputs must reference earlier scratch slots.");
          }
        }
        Semantics::validateSurfaceDial(model_, dial, model_.compositeInputs.size());
        validateFormulaDial(model_, dial, model_.compositeInputs.size());
      }
    }
    for( const auto& dial : model_.dials ){
      if( dial.type != BackendDialType::Composite ){ continue; }
      if( dial.compositeProgram >= model_.compositePrograms.size()
          or dial.inputCount != model_.compositePrograms[dial.compositeProgram].inputCount
          or dial.firstInput > model_.dialInputs.size()
          or dial.inputCount > model_.dialInputs.size() - dial.firstInput ){
        throw std::runtime_error("Invalid CPU composite root.");
      }
    }
  }

  // The caller owns scratch (one preallocated buffer per worker). Instruction
  // inputs refer only to graph arguments or previously computed results.
  inline double evaluateComposite(const PropagationView& model_, const BackendDialDescriptor& root_,
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
          value = std::abs(std::fmod(value - input.mirrorMin, 2 * input.mirrorRange));
          if( value > input.mirrorRange ){ value = 2 * input.mirrorRange - value; }
          value += input.mirrorMin;
        }
        return value;
      };
      const auto* data = dial.payloadSize ? model_.dialPayloads.data() + dial.payloadOffset : nullptr;
      double response;
      if( dial.type == BackendDialType::Shift ){ response = data[0]; }
      else if( dial.type == BackendDialType::Formula ){
        const auto read = [&](std::size_t index_) {
          const auto& binding = model_.formulaInputs[dial.firstFormulaInput + index_];
          return binding.parameterIndex < 0 ? binding.constant : load(binding.parameterIndex);
        };
        response = evaluateFormula(model_.formulaPrograms[dial.formulaProgramIndex], read, formulaScratch_);
      }
      else{
        double x = load(0);
        if( dial.type == BackendDialType::Norm ){ response = x; }
        else if( dial.type == BackendDialType::Bilinear or dial.type == BackendDialType::Bicubic ){
          double y = load(1);
          const int nx = int(data[0]), ny = int(data[1]);
          const auto* xx = data + 2;
          const auto* yy = xx + nx;
          const auto* knots = yy + ny;
          if( not dial.allowExtrapolation ){
            if( x < xx[0] ){ x = xx[0]; }
            if( x > xx[nx - 1] ){ x = xx[nx - 1]; }
            if( y < yy[0] ){ y = yy[0]; }
            if( y > yy[ny - 1] ){ y = yy[ny - 1]; }
          }
          response = dial.type == BackendDialType::Bilinear
              ? CalculateBilinearInterpolation(x, y, -1E20, 1E20, knots, nx, ny, xx, nx, yy, ny)
              : CalculateBicubicSpline(x, y, -1E20, 1E20, knots, nx, ny, xx, nx, yy, ny);
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
          switch( dial.type ){
            case BackendDialType::CompactSpline:
              response = CalculateCompactSpline(x, -1E20, 1E20, data, int(dial.payloadSize - 2)); break;
            case BackendDialType::UniformSpline:
              response = CalculateUniformSpline(x, -1E20, 1E20, data, int(dial.payloadSize)); break;
            case BackendDialType::MonotonicSpline:
              response = CalculateMonotonicSpline(x, -1E20, 1E20, data, int(dial.payloadSize - 2)); break;
            case BackendDialType::GeneralSpline:
              response = CalculateGeneralSpline(x, -1E20, 1E20, data, int(dial.payloadSize)); break;
            default: throw std::runtime_error("Invalid CPU composite instruction.");
          }
        }
      }
      scratch_[program.inputCount + i] = Semantics::clampDialResponse(dial, response);
    }
    return Semantics::clampDialResponse(root_, scratch_[program.outputSlot]);
  }
}
#endif
