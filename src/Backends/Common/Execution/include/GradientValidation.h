#ifndef GUNDAM_GRADIENT_VALIDATION_H
#define GUNDAM_GRADIENT_VALIDATION_H
#include "DialProgramValidation.h"

#include <cmath>
#include <stdexcept>

namespace Backends {
  // Structural physics rules shared by CPU, MPS and CUDA. Executors retain
  // lifecycle checks, producer-publication checks and device storage limits.
  inline void validateGradientModel(const EngineView& view_) {
    const auto& m = view_.propagation;
    for( const auto& sample : m.samples ){
      if( sample.useDynamicBinning and not sample.binning.isSmoothed )
        throw std::runtime_error("Backend dynamic binning gradients require bin smoothing.");
    }
    for( const auto& event : m.events ){
      for( std::size_t i = 0; i < event.variableResponseCount; ++i ){
        const auto& target = m.variableResponses.at(event.firstVariableResponse+i);
        const auto& sample = m.samples.at(event.samplePosition);
        const auto column = target.variableIndex-event.firstVariable;
        if( column >= sample.binning.variableNames.size() ) continue;
        if( not sample.useDynamicBinning or not sample.binning.isSmoothed )
          throw std::runtime_error("Backend observable gradients require dynamic binning and bin smoothing.");
        for( const auto& interval : sample.binning.intervals ){
          if( interval.variableIndex == column and
              (interval.lowerInverseWidth <= 0. or interval.upperInverseWidth <= 0.) )
            throw std::runtime_error("Backend observable gradients require bin smoothing for every moving axis: "
                                     + sample.binning.variableNames[column]);
        }
      }
    }
    if( view_.likelihood.empty() ) throw std::runtime_error("Backend gradient requires a statistical likelihood.");
    for( const auto& sample : view_.likelihood.samples )
      if( sample.portableOptions.kind != LikelihoodKind::BarlowBeestonLiteGaussian )
        throw std::runtime_error("Backend gradients currently require BarlowBeestonLiteGaussian.");
    validateCompositePrograms(m);
    for( const auto& program : m.formulaPrograms ) program.validate();
    for( const auto& dial : m.dials ){
      switch( dial.type ){
        case BackendDialType::Norm: case BackendDialType::Graph:
        case BackendDialType::CompactSpline: case BackendDialType::UniformSpline:
        case BackendDialType::MonotonicSpline: case BackendDialType::GeneralSpline:
          if( dial.inputCount != 1 ) throw std::runtime_error("Backend scalar gradient requires one input.");
          break;
        case BackendDialType::Bilinear: case BackendDialType::Bicubic: Semantics::validateSurfaceDial(m,dial); break;
        case BackendDialType::Formula: validateFormulaDial(m,dial); break;
        case BackendDialType::Tabulated: case BackendDialType::Kriged: Semantics::validateTableDial(m,dial); break;
        case BackendDialType::Shift: case BackendDialType::Composite: case BackendDialType::ExternalWeight: break;
        default: throw std::runtime_error("Unsupported backend gradient dial.");
      }
      for( std::size_t j = 0; j < dial.inputCount; ++j ){
        const auto& input = m.dialInputs.at(dial.firstInput+j);
        if( input.source != DialInputSource::Parameter || input.parameterIndex >= m.parameterCount )
          throw std::runtime_error("Backend gradient requires valid parameter inputs.");
      }
    }
    const auto mirror = [](const auto& input_){
      if( input_.useMirror && (!std::isfinite(input_.mirrorRange) || input_.mirrorRange <= 0.) )
        throw std::runtime_error("Backend gradient requires positive finite mirror ranges.");
    };
    for( const auto& input : m.dialInputs ) mirror(input);
    for( const auto& input : m.compositeInputs ) mirror(input);
  }
}
#endif
