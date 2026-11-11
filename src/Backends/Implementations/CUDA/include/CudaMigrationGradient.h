#ifndef GUNDAM_CUDA_MIGRATION_GRADIENT_H
#define GUNDAM_CUDA_MIGRATION_GRADIENT_H
#include "CudaGradientSemantics.h"
#include "CudaBinning.h"
#include "GradientTypes.h"

namespace Backends::CudaGradient {
  struct EventGradientView {
    const DeviceIndex* indices;
    const DeviceIndex* inputOffsets;
    const DeviceIndex* eventOffsets;
    const EventView* events;
    const BackendVariableResponseDescriptor* targets;
    const std::uint32_t* variableDials;
    const CudaBinningSample* samples;
    std::size_t sampleCount;
    const BinningBin* bins;
    const BinningInterval* intervals;
    const CudaBinningNode* nodes;
    const CudaBinningBound* bounds;
    const double* variables;
  };

  // value = LLH adjoint of the event weight; slope = LLH adjoint of
  // one histogram coordinate. Reuse the value path's candidate traversal,
  // compact-support fractions and first-largest roundoff correction.
  GUNDAM_BACKEND_HOST_DEVICE inline Dual binAdjoints(EventGradientView view_, const EventView& event_,
      const CudaBinningSample& sample_, std::size_t column_, double weight_,
      const GradientSemantics::DialEvaluation<double>* scales_) {
    const double* values = view_.variables+event_.firstVariable;
    Semantics::CudaBinCandidates candidates(sample_,values,view_.nodes,view_.bounds);
    Dual result;
    double total = 0., totalSlope = 0., largest = -1., largestSlope = 0.;
    GradientSemantics::DialEvaluation<double> largestScales{};
    for( auto j = candidates.next(); j < sample_.endBin; j = candidates.next() ){
      const auto& bin = view_.bins[j];
      Dual fraction(1.);
      for( std::size_t k = 0; k < bin.intervalCount; ++k ){
        const auto& interval = view_.intervals[bin.firstInterval+k];
        const double value = values[interval.variableIndex];
        const double f = GundamBinningMath::intervalFraction(interval.lowerBound,interval.upperBound,
            interval.lowerInverseWidth,interval.upperInverseWidth,value);
        double df = 0.;
        if( f > 0. && interval.variableIndex == column_ && interval.lowerInverseWidth > 0. )
          df = GundamBinningMath::cumulativeKernelSlope((interval.lowerBound-value)*interval.lowerInverseWidth)*interval.lowerInverseWidth
             - GundamBinningMath::cumulativeKernelSlope((interval.upperBound-value)*interval.upperInverseWidth)*interval.upperInverseWidth;
        fraction = fraction*Dual(f,df);
        if( fraction.value == 0. ) break;
      }
      if( fraction.value == 0. ) continue;
      const auto scale = scales_[sample_.binOffset+bin.binIndex];
      const double adjoint = scale.value+weight_*fraction.value*scale.derivative;
      result.value += adjoint*fraction.value;
      result.slope += adjoint*weight_*fraction.slope;
      total += fraction.value; totalSlope += fraction.slope;
      if( fraction.value > largest ){
        largest = fraction.value; largestSlope = fraction.slope; largestScales = scale;
      }
    }
    if( total > 1. ){
      const double corrected = largest-(total-1.);
      const double rawScale = largestScales.value+weight_*largest*largestScales.derivative;
      const double correctedScale = largestScales.value+weight_*corrected*largestScales.derivative;
      result.value += correctedScale*corrected-rawScale*largest;
      result.slope += weight_*(correctedScale*(largestSlope-totalSlope)-rawScale*largestSlope);
    }
    return result;
  }
  GUNDAM_BACKEND_HOST_DEVICE inline void writeDial(double scale_, std::size_t dial_, std::size_t& offset_,
      const DeviceIndex* offsets_, const GradientSemantics::DialEvaluation<double>* responses_, double* out_) {
    for( auto k = offsets_[dial_]; k < offsets_[dial_+1]; ++k )
      out_[offset_++] = scale_ == 0. ? 0. : scale_*responses_[k].derivative;
  }
  GUNDAM_BACKEND_HOST_DEVICE inline void eventGradient(std::size_t i_, EventGradientView view_,
      GradientEvent<double> event_, GradientWork<double> work_,
      const GradientSemantics::DialEvaluation<double>* responses_,
      const GradientSemantics::DialEvaluation<double>* scales_, const double* weights_, double* out_) {
    const auto& original = view_.events[i_];
    const auto* sample = original.samplePosition < view_.sampleCount ? view_.samples+original.samplePosition : nullptr;
    const bool smoothed = sample && sample->isSmoothed;
    const double weight = weights_[event_.result];
    auto offset = std::size_t(view_.eventOffsets[i_]);
    double prefix = 1.;
    for( std::size_t j = 0; j < event_.count; ++j ){
      auto dial = view_.indices[event_.first+j];
      for( auto k = view_.inputOffsets[dial]; k < view_.inputOffsets[dial+1]; ++k ) out_[offset++] = prefix;
      prefix *= responses_[view_.inputOffsets[dial]].value;
    }
    const auto observableOffset = offset;
    double scale = 0.;
    if( !(work_.useCap && prefix > work_.cap) ){
      if( smoothed ) scale = binAdjoints(view_,original,*sample,std::size_t(-1),weight,scales_).value;
      else if( event_.bin < work_.bins ) scale = scales_[event_.bin].value+weight*scales_[event_.bin].derivative;
      scale *= event_.base;
    }
    double suffix = 1.;
    for( std::size_t j = event_.count; j-- > 0; ){
      auto dial = view_.indices[event_.first+j];
      for( auto k = view_.inputOffsets[dial+1]; k-- > view_.inputOffsets[dial]; ){
        --offset; out_[offset] = scale == 0. ? 0. : scale*out_[offset]*suffix*responses_[k].derivative;
      }
      suffix *= responses_[view_.inputOffsets[dial]].value;
    }
    // Caps suppress only weight derivatives; migration retains the capped w.
    offset = observableOffset;
    for( std::size_t j = 0; j < original.variableResponseCount; ++j ){
      const auto& target = view_.targets[original.firstVariableResponse+j];
      auto column = target.variableIndex-original.firstVariable;
      const double adjoint = smoothed && column < sample->variableCount
          ? binAdjoints(view_,original,*sample,column,weight,scales_).slope : 0.;
      const bool hasOverride = target.overrideDial != std::size_t(-1);
      auto overrideOffset = offset;
      if( hasOverride ) offset += view_.inputOffsets[target.overrideDial+1]-view_.inputOffsets[target.overrideDial];
      double product = hasOverride ? responses_[view_.inputOffsets[target.overrideDial]].value : target.referenceValue;
      for( std::size_t k = 0; k < target.multiplyDialCount; ++k ){
        auto dial = view_.variableDials[target.firstMultiplyDial+k];
        for( auto d = view_.inputOffsets[dial]; d < view_.inputOffsets[dial+1]; ++d ) out_[offset++] = product;
        product *= responses_[view_.inputOffsets[dial]].value;
      }
      const auto afterMultiply = offset;
      double trailing = 1.;
      for( std::size_t k = target.multiplyDialCount; k-- > 0; ){
        auto dial = view_.variableDials[target.firstMultiplyDial+k];
        for( auto d = view_.inputOffsets[dial+1]; d-- > view_.inputOffsets[dial]; ){
          --offset; out_[offset] = adjoint == 0. ? 0. : adjoint*out_[offset]*trailing*responses_[d].derivative;
        }
        trailing *= responses_[view_.inputOffsets[dial]].value;
      }
      if( hasOverride ) writeDial(adjoint*trailing,target.overrideDial,overrideOffset,view_.inputOffsets,responses_,out_);
      offset = afterMultiply;
      for( std::size_t k = 0; k < target.addDialCount; ++k )
        writeDial(adjoint,view_.variableDials[target.firstAddDial+k],offset,view_.inputOffsets,responses_,out_);
    }
  }
}
#endif
