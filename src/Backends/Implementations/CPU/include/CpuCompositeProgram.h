#ifndef GUNDAM_CPU_COMPOSITE_PROGRAM_H
#define GUNDAM_CPU_COMPOSITE_PROGRAM_H

#include "EngineView.h"
#include "NativeDialProgramModel.h"
#include "CpuFormulaProgram.h"
#include "Semantics/NativeCompositeProgram.h"
#include "Semantics/BackendDialSemantics.h"
#include "CalculateBilinearInterpolation.h"
#include "CalculateBicubicSpline.h"

#include <cmath>

namespace Backends {
  namespace Detail {
    struct CpuCompositeOperations {
      static constexpr bool checkSplineInput = false;
      static constexpr bool packed = false;
      static GUNDAM_BACKEND_ALWAYS_INLINE const CompositeProgram& program(const PropagationView& model_, const BackendDialDescriptor& root_) {
        return model_.compositePrograms[root_.compositeProgram];
      }
      static GUNDAM_BACKEND_ALWAYS_INLINE std::size_t formulaOffset(const PropagationView&, const BackendDialDescriptor&,
          std::size_t, const BackendDialDescriptor& dial_) { return dial_.firstFormulaInput; }
      static GUNDAM_BACKEND_ALWAYS_INLINE const double* payload(const PropagationView& model_, const BackendDialDescriptor&,
          std::size_t, const BackendDialDescriptor& dial_) {
        return dial_.payloadSize ? model_.dialPayloads.data() + dial_.payloadOffset : nullptr;
      }
      static GUNDAM_BACKEND_ALWAYS_INLINE const double* payloads(const PropagationView& model_) {
        return model_.dialPayloads.data();
      }
      template<typename InputReader>
      static GUNDAM_BACKEND_ALWAYS_INLINE double formula(const PropagationView& model_,
          const BackendDialDescriptor& dial_, InputReader read_, double* stack_) {
        return evaluateBoundFormula(model_.formulaPrograms[dial_.formulaProgramIndex], read_, stack_);
      }
      static GUNDAM_BACKEND_ALWAYS_INLINE double surface(const BackendDialDescriptor& dial_,
          double x_, double y_, const double* data_) {
        const int nx = int(data_[0]), ny = int(data_[1]);
        const auto* xx = data_ + 2;
        const auto* yy = xx + nx;
        const auto* knots = yy + ny;
        if( not dial_.allowExtrapolation ){
          if( x_ < xx[0] ){ x_ = xx[0]; }
          if( x_ > xx[nx - 1] ){ x_ = xx[nx - 1]; }
          if( y_ < yy[0] ){ y_ = yy[0]; }
          if( y_ > yy[ny - 1] ){ y_ = yy[ny - 1]; }
        }
        return dial_.type == BackendDialType::Bilinear
            ? CalculateBilinearInterpolation(x_, y_, -1E20, 1E20, knots, nx, ny, xx, nx, yy, ny)
            : CalculateBicubicSpline(x_, y_, -1E20, 1E20, knots, nx, ny, xx, nx, yy, ny);
      }
      static inline double invalidOpcode() {
        throw std::runtime_error("Invalid CPU composite instruction.");
      }
    };
  }

  namespace Detail {
    struct CpuPackedCompositeOperations : CpuCompositeOperations {
      template<typename InputReader>
      static GUNDAM_BACKEND_ALWAYS_INLINE double fastFormula(const NativeDialProgramModel& model_,
          const BackendDialDescriptor& dial_, InputReader read_) {
        return evaluateNativeFormulaPlan(model_.formulaPrograms[dial_.formulaProgramIndex].nativePlan, read_);
      }
      static constexpr bool packed = true;
      static GUNDAM_BACKEND_ALWAYS_INLINE const NativeCompositeTemplate& program(const NativeDialProgramModel& model_, const BackendDialDescriptor& root_) {
        return model_.compositePrograms[model_.compositeInstances[root_.compositeProgram].programIndex];
      }
      static GUNDAM_BACKEND_ALWAYS_INLINE std::size_t formulaOffset(const NativeDialProgramModel& model_, const BackendDialDescriptor& root_,
          std::size_t node_, const BackendDialDescriptor&) {
        return model_.compositeNodeData[model_.compositeInstances[root_.compositeProgram].firstNodeData + node_];
      }
      static GUNDAM_BACKEND_ALWAYS_INLINE const double* payload(const NativeDialProgramModel& model_, const BackendDialDescriptor& root_,
          std::size_t node_, const BackendDialDescriptor& dial_) {
        return dial_.payloadSize ? model_.payloads + formulaOffset(model_, root_, node_, dial_) : nullptr;
      }
      template<typename InputReader>
      static GUNDAM_BACKEND_ALWAYS_INLINE double formula(const NativeDialProgramModel& model_,
          const BackendDialDescriptor& dial_, InputReader read_, double* stack_) {
        return evaluateBoundFormula(model_.formulaPrograms[dial_.formulaProgramIndex], read_, stack_);
      }
    };
  }
  inline double evaluatePackedComposite(const NativeDialProgramModel& model_, const BackendDialDescriptor& root_,
      const double* parameters_, double* scratch_, double* formulaScratch_) {
    return evaluateNativeComposite<Detail::CpuPackedCompositeOperations>(model_, root_, parameters_, scratch_, formulaScratch_);
  }
  inline double evaluatePackedFormula(const NativeDialProgramModel& model_, const BackendDialDescriptor& dial_,
      std::size_t index_, const double* parameters_, double* scratch_) {
    const auto data = model_.rootFormulaData[index_];
    const auto read = [&](std::size_t i_) {
      const auto& binding = model_.formulaInputs[data.second + i_];
      if( binding.parameterIndex < 0 ){ return binding.constant; }
      const auto& input = model_.dialInputs[dial_.firstInput + binding.parameterIndex];
      return Semantics::transformDialInput(input, parameters_[input.parameterIndex]);
    };
    return Semantics::clampDialResponse(dial_, evaluateBoundFormula(model_.formulaPrograms[data.first], read, scratch_));
  }

  // The caller owns one preallocated scratch buffer per worker. Adapters have
  // no runtime state, preserving the original CPU interpreter's argument count.
  inline double evaluateComposite(const PropagationView& model_, const BackendDialDescriptor& root_,
                                  const double* parameters_, double* scratch_, double* formulaScratch_) {
    return evaluateNativeComposite<Detail::CpuCompositeOperations>(
        model_, root_, parameters_, scratch_, formulaScratch_);
  }
}
#endif
