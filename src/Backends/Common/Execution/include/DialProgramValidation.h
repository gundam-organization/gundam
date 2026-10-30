#ifndef GUNDAM_DIAL_PROGRAM_VALIDATION_H
#define GUNDAM_DIAL_PROGRAM_VALIDATION_H
#include "EngineView.h"
#include "Semantics/BackendDialSemantics.h"
#include <stdexcept>
namespace Backends {
  inline void validateFormulaDial(const PropagationView& model_, const BackendDialDescriptor& dial_,
                                   std::size_t availableInputs_ = std::size_t(-1)) {
    if( dial_.type != BackendDialType::Formula ){ return; }
    if( availableInputs_ == std::size_t(-1) ){ availableInputs_ = model_.dialInputs.size(); }
    if( dial_.formulaProgramIndex >= model_.formulaPrograms.size()
        or dial_.firstInput > availableInputs_
        or dial_.inputCount > availableInputs_ - dial_.firstInput ){
      throw std::runtime_error("Invalid backend formula dial descriptor");
    }
    const auto count = model_.formulaPrograms[dial_.formulaProgramIndex].inputCount;
    if( dial_.firstFormulaInput > model_.formulaInputs.size()
        or count > model_.formulaInputs.size() - dial_.firstFormulaInput ){
      throw std::runtime_error("Invalid backend formula input range");
    }
    for( std::size_t i = 0; i < count; ++i ){
      const auto index = model_.formulaInputs[dial_.firstFormulaInput + i].parameterIndex;
      if( index < -1 or (index >= 0 and std::size_t(index) >= dial_.inputCount) ){
        throw std::runtime_error("Invalid backend formula input binding");
      }
    }
  }

  inline void validateCompositePrograms(const PropagationView& model_) {
    for( const auto& program : model_.compositePrograms ){
      if( program.firstInstruction > model_.compositeInstructions.size()
          or program.instructionCount > model_.compositeInstructions.size() - program.firstInstruction
          or program.instructionCount > std::size_t(-1) - program.inputCount
          or program.outputSlot >= program.scratchSize() ){
        throw std::runtime_error("Invalid backend composite program range.");
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
          default: throw std::runtime_error("Unsupported backend composite instruction type.");
        }
        if( dial.firstInput > model_.compositeInputs.size()
            or dial.inputCount > model_.compositeInputs.size() - dial.firstInput
            or (dial.type != BackendDialType::Shift and dial.type != BackendDialType::Formula
                and dial.inputCount != (surface ? 2u : 1u))
            or dial.payloadOffset > model_.dialPayloads.size()
            or dial.payloadSize > model_.dialPayloads.size() - dial.payloadOffset
            or dial.payloadSize < minimumPayload ){
          throw std::runtime_error("Invalid backend composite instruction layout.");
        }
        for( std::size_t j = 0; j < dial.inputCount; ++j ){
          const auto& input = model_.compositeInputs[dial.firstInput + j];
          if( input.source != DialInputSource::Parameter or input.parameterIndex >= program.inputCount + i ){
            throw std::runtime_error("backend composite inputs must reference earlier scratch slots.");
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
        throw std::runtime_error("Invalid backend composite root.");
      }
    }
  }

}
#endif
