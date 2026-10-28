#ifndef GUNDAM_CPU_COMPOSITE_PROGRAM_H
#define GUNDAM_CPU_COMPOSITE_PROGRAM_H

#include "EngineView.h"
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

  // The caller owns one preallocated scratch buffer per worker. Adapters have
  // no runtime state, preserving the original CPU interpreter's argument count.
  inline double evaluateComposite(const PropagationView& model_, const BackendDialDescriptor& root_,
                                  const double* parameters_, double* scratch_, double* formulaScratch_) {
    return evaluateNativeComposite<Detail::CpuCompositeOperations>(
        model_, root_, parameters_, scratch_, formulaScratch_);
  }
}
#endif
