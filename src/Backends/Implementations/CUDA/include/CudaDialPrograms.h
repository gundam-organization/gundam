#ifndef GUNDAM_CUDA_DIAL_PROGRAMS_H
#define GUNDAM_CUDA_DIAL_PROGRAMS_H

#include "CudaWeightModel.h"
#include "Semantics/BackendDialSemanticsCore.h"
#include "Semantics/NativeFormulaProgram.h"
#include "Semantics/NativeCompositeProgram.h"

#include <cmath>

namespace Backends {
  // Non-owning, trivially copyable kernel argument; every pointer addresses device memory.
  struct CudaProgramView {
    const CudaFormulaProgram* formulaPrograms{};
    const FormulaInstruction* formulaInstructions{};
    const double* formulaConstants{};
    const FormulaInput* formulaInputs{};
    const NativeCompositeTemplate* compositePrograms{};
    const CompositeInstruction* compositeInstructions{};
    const BackendDialInputDescriptor* compositeInputs{};
    const BackendDialInputDescriptor* dialInputs{};
    const double* payloads{};
    const std::size_t* scratchOffsets{};
    double* scratch{};
    const NativeCompositeInstance* compositeInstances{};
    const std::size_t* compositeNodeData{};
  };

  template<typename InputReader>
  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE double evaluateCudaFormula(
      const CudaFormulaProgram& program_, const FormulaInstruction* instructions_,
      const double* constants_, InputReader readInput_, double* stack_) {
    const auto* first = instructions_ + program_.firstInstruction;
    const NativeFormulaView program{{first, first + program_.instructionCount},
        constants_ ? constants_ + program_.firstConstant : nullptr, program_.nativePlan};
    return evaluateNativeFormula(program, readInput_, stack_, []() -> double {
          GUNDAM_BACKEND_SEMANTICS_ASSERT(false);
          return NAN;
        });
  }

  namespace Detail {
    struct CudaCompositeOperations {
      template<typename InputReader>
      static GUNDAM_BACKEND_ALWAYS_INLINE GUNDAM_BACKEND_HOST_DEVICE
      double fastFormula(const CudaProgramView& model_, const BackendDialDescriptor& dial_, InputReader read_) {
        return evaluateNativeFormulaPlan(model_.formulaPrograms[dial_.formulaProgramIndex].nativePlan, read_);
      }
      static constexpr bool checkSplineInput = true;
      static constexpr bool packed = true;
      static GUNDAM_BACKEND_ALWAYS_INLINE GUNDAM_BACKEND_HOST_DEVICE
      const NativeCompositeTemplate& program(const CudaProgramView& model_, const BackendDialDescriptor& root_) {
        return model_.compositePrograms[model_.compositeInstances[root_.compositeProgram].programIndex];
      }
      static GUNDAM_BACKEND_ALWAYS_INLINE GUNDAM_BACKEND_HOST_DEVICE
      std::size_t formulaOffset(const CudaProgramView& model_, const BackendDialDescriptor& root_,
          std::size_t node_, const BackendDialDescriptor&) {
        return model_.compositeNodeData[model_.compositeInstances[root_.compositeProgram].firstNodeData + node_];
      }
      static GUNDAM_BACKEND_ALWAYS_INLINE GUNDAM_BACKEND_HOST_DEVICE
      const double* payload(const CudaProgramView& model_, const BackendDialDescriptor& root_,
          std::size_t node_, const BackendDialDescriptor& dial_) {
        return dial_.payloadSize ? model_.payloads + formulaOffset(model_, root_, node_, dial_) : nullptr;
      }
      static GUNDAM_BACKEND_ALWAYS_INLINE GUNDAM_BACKEND_HOST_DEVICE
      const double* payloads(const CudaProgramView& model_) { return model_.payloads; }
      template<typename InputReader>
      static GUNDAM_BACKEND_ALWAYS_INLINE GUNDAM_BACKEND_HOST_DEVICE
      double formula(const CudaProgramView& model_, const BackendDialDescriptor& dial_,
                     InputReader read_, double* stack_) {
        return evaluateCudaFormula(model_.formulaPrograms[dial_.formulaProgramIndex],
            model_.formulaInstructions, model_.formulaConstants, read_, stack_);
      }
      static GUNDAM_BACKEND_ALWAYS_INLINE GUNDAM_BACKEND_HOST_DEVICE
      double surface(const BackendDialDescriptor& dial_, double x_, double y_, const double* data_) {
        return GundamDeviceMath::EvaluateSurface(x_, y_, data_,
            dial_.type == BackendDialType::Bicubic, dial_.allowExtrapolation);
      }
      static GUNDAM_BACKEND_ALWAYS_INLINE GUNDAM_BACKEND_HOST_DEVICE
      double invalidOpcode() {
        GUNDAM_BACKEND_SEMANTICS_ASSERT(false);
        return NAN;
      }
    };
  }

  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE double evaluateCudaComposite(const CudaProgramView& model_,
      const BackendDialDescriptor& root_, const double* parameters_, double* scratch_, double* formulaScratch_) {
    return evaluateNativeComposite<Detail::CudaCompositeOperations>(
        model_, root_, parameters_, scratch_, formulaScratch_);
  }

  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double evaluateCudaFastProgram(const CudaProgramView& view_, const BackendDialDescriptor& dial_, const double* parameters_) {
    if( dial_.type == BackendDialType::Composite ){
      return evaluateNativeCompositeChain<Detail::CudaCompositeOperations>(view_, dial_, parameters_);
    }
    const auto read = [&](std::size_t i_) {
      const auto& binding = view_.formulaInputs[dial_.firstFormulaInput + i_];
      if( binding.parameterIndex < 0 ){ return binding.constant; }
      const auto& input = view_.dialInputs[dial_.firstInput + binding.parameterIndex];
      return Semantics::transformDialInput(input, parameters_[input.parameterIndex]);
    };
    return Semantics::clampDialResponse(dial_, Detail::CudaCompositeOperations::fastFormula(view_, dial_, read));
  }

  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE double evaluateCudaProgram(
      const CudaProgramView& view_, const BackendDialDescriptor& dial_,
      std::size_t index_, const double* parameters_) {
    if( dial_.type == BackendDialType::Composite ){
      const auto& program = Detail::CudaCompositeOperations::program(view_, dial_);
      if( program.isChain ){ return evaluateCudaComposite(view_, dial_, parameters_, nullptr, nullptr); }
      auto* scratch = view_.scratch + view_.scratchOffsets[index_];
      return evaluateCudaComposite(view_, dial_, parameters_, scratch,
                                   scratch + program.inputCount + program.instructionCount);
    }
    const auto& formula = view_.formulaPrograms[dial_.formulaProgramIndex];
    if( formula.nativePlan.kernel != NativeFormulaKernel::Interpreter ){
      return evaluateCudaFastProgram(view_, dial_, parameters_);
    }
    auto* scratch = view_.scratch + view_.scratchOffsets[index_];
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
