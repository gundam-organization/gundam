#ifndef GUNDAM_NATIVE_COMPOSITE_PROGRAM_H
#define GUNDAM_NATIVE_COMPOSITE_PROGRAM_H
#include "Semantics/BackendDialSemanticsCore.h"
namespace Backends {
  GUNDAM_BACKEND_ALWAYS_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double mirrorCompositeInput(const BackendDialInputDescriptor& input_, double value_) {
    if( not input_.useMirror ){ return value_; }
    value_ = ::fabs(::fmod(value_ - input_.mirrorMin, 2 * input_.mirrorRange));
    if( value_ > input_.mirrorRange ){ value_ = 2 * input_.mirrorRange - value_; }
    return value_ + input_.mirrorMin;
  }
  template<bool scalarOnly_, typename Operations, typename SecondReader>
  GUNDAM_BACKEND_ALWAYS_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double evaluateNativeCompositeLeaf(const CompositeInstruction& instruction, double x,
      const double* data, SecondReader readSecond_, bool& valid_) {
    const auto& dial = instruction.dial;

    if( dial.type == BackendDialType::Norm ){ return x; }
    else if( dial.type == BackendDialType::Bilinear or dial.type == BackendDialType::Bicubic ){
      if constexpr( scalarOnly_ ){ return Operations::invalidOpcode(); }
      else{ return Operations::surface(dial, x, readSecond_(), data); }
    }
    else if( dial.type == BackendDialType::Graph ){
      // Graph returns endpoint ordinates directly when extrapolation is off.
      if( not dial.allowExtrapolation and x <= data[1] ){ return data[0]; }
      else if( not dial.allowExtrapolation and x >= data[dial.payloadSize - 1] ){ return data[dial.payloadSize - 2]; }
      else{ return CalculateGraph(x, -1E20, 1E20, data, int(dial.payloadSize)); }
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
          if( not (position >= -limit and position <= limit) ){ valid_ = false; return NAN; }
        }
      }
      switch( dial.type ){
        case BackendDialType::CompactSpline:
          return CalculateCompactSpline(x, -1E20, 1E20, data, int(dial.payloadSize - 2));
        case BackendDialType::UniformSpline:
          return CalculateUniformSpline(x, -1E20, 1E20, data, int(dial.payloadSize));
        case BackendDialType::MonotonicSpline:
          return CalculateMonotonicSpline(x, -1E20, 1E20, data, int(dial.payloadSize - 2));
        case BackendDialType::GeneralSpline:
          return CalculateGeneralSpline(x, -1E20, 1E20, data, int(dial.payloadSize));
        default: return Operations::invalidOpcode();
      }
    }
  }
  // Three-node chains keep graph values and the fast formula intermediates in
  // scalar variables. No globally indexed graph/bytecode stack is touched.
  template<typename Operations, typename Model>
  GUNDAM_BACKEND_ALWAYS_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double evaluateNativeCompositeChain(const Model& model_, const BackendDialDescriptor& root_, const double* parameters_) {
    const auto& program = Operations::program(model_, root_);
    const auto& first = model_.compositeInstructions[program.firstInstruction];
    const auto& second = model_.compositeInstructions[program.firstInstruction + 1];
    const auto& third = model_.compositeInstructions[program.firstInstruction + 2];
    const auto& xInput = model_.dialInputs[root_.firstInput];
    const auto& yInput = model_.dialInputs[root_.firstInput + 1];
    const double x = Semantics::transformDialInput(xInput, parameters_[xInput.parameterIndex]);
    const double y = Semantics::transformDialInput(yInput, parameters_[yInput.parameterIndex]);
    const auto firstOffset = Operations::formulaOffset(model_, root_, 0, first.dial);
    const auto readFirst = [&](std::size_t i_) {
      const auto& binding = model_.formulaInputs[firstOffset + i_];
      if( binding.parameterIndex < 0 ){ return binding.constant; }
      const auto& input = model_.compositeInputs[first.dial.firstInput + binding.parameterIndex];
      return mirrorCompositeInput(input, input.parameterIndex == 0 ? x : y);
    };
    double value = Semantics::clampDialResponse(first.dial, Operations::fastFormula(model_, first.dial, readFirst));
    const auto& leafInput = model_.compositeInputs[second.dial.firstInput];
    value = mirrorCompositeInput(leafInput, value);
    bool valid = true;
    value = evaluateNativeCompositeLeaf<true, Operations>(second, value,
        Operations::payload(model_, root_, 1, second.dial), [](){ return 0.; }, valid);
    if constexpr( Operations::checkSplineInput ){ if( not valid ){ return NAN; } }
    value = Semantics::clampDialResponse(second.dial, value);
    value = mirrorCompositeInput(model_.compositeInputs[third.dial.firstInput], value);
    const auto thirdOffset = Operations::formulaOffset(model_, root_, 2, third.dial);
    const auto readThird = [&](std::size_t i_) {
      const auto& binding = model_.formulaInputs[thirdOffset + i_];
      return binding.parameterIndex < 0 ? binding.constant : value;
    };
    value = Operations::fastFormula(model_, third.dial, readThird);
    return Semantics::clampDialResponse(root_, Semantics::clampDialResponse(third.dial, value));
  }
  template<typename Operations, typename Model>
  GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
  double evaluateNativeComposite(const Model& model_, const BackendDialDescriptor& root_,
      const double* parameters_, double* scratch_, double* formulaScratch_) {
    if constexpr( Operations::packed ){
      if( Operations::program(model_, root_).isChain ){
        return evaluateNativeCompositeChain<Operations>(model_, root_, parameters_);
      }
    }
    const auto& program = Operations::program(model_, root_);
    for( std::size_t i = 0; i < program.inputCount; ++i ){
      const auto& input = model_.dialInputs[root_.firstInput + i];
      scratch_[i] = Semantics::transformDialInput(input, parameters_[input.parameterIndex]);
    }
    for( std::size_t i = 0; i < program.instructionCount; ++i ){
      const auto& instruction = model_.compositeInstructions[program.firstInstruction + i];
      const auto& dial = instruction.dial;
      const auto load = [&](std::size_t index_) {
        const auto& input = model_.compositeInputs[dial.firstInput + index_];
        return mirrorCompositeInput(input, scratch_[input.parameterIndex]);
      };
      double response;
      if( dial.type == BackendDialType::Formula ){
        const auto offset = Operations::formulaOffset(model_, root_, i, dial);
        const auto read = [&](std::size_t index_) {
          const auto& binding = model_.formulaInputs[offset + index_];
          return binding.parameterIndex < 0 ? binding.constant : load(binding.parameterIndex);
        };
        response = Operations::formula(model_, dial, read, formulaScratch_);
      }
      else{
        const auto* data = Operations::payload(model_, root_, i, dial);
        if( dial.type == BackendDialType::Shift ){ response = data[0]; }
        else{
          bool valid = true;
          response = evaluateNativeCompositeLeaf<false, Operations>(instruction, load(0), data, [&](){ return load(1); }, valid);
          if constexpr( Operations::checkSplineInput ){ if( not valid ){ return NAN; } }
        }
      }
      scratch_[program.inputCount + i] = Semantics::clampDialResponse(dial, response);
    }
    return Semantics::clampDialResponse(root_, scratch_[program.outputSlot]);
  }
}
#endif
