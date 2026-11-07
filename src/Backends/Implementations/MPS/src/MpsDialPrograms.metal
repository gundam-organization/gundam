using namespace Backends;

ObservableScalar mirror_program_input(ObservableScalar value, uint useMirror,
                                      ObservableScalar minimum, ObservableScalar range) {
  if( !useMirror ){ return value; }
  value = GundamDeviceMath::remainder(value - minimum, ObservableScalar(2) * range);
  if( value < ObservableScalar(0) ){ value = -value; }
  if( value > range ){ value = ObservableScalar(2) * range - value; }
  return value + minimum;
}

ObservableScalar read_program_input(DeviceProgramInput input,
    device const ObservableScalar* parameters, device const ObservableScalar* scratch) {
  if( input.source == ProgramInputSource::Constant ){ return input.value; }
  auto value = input.source == ProgramInputSource::Parameter ? parameters[input.index] : scratch[input.index];
  return mirror_program_input(value, input.useMirror, input.mirrorMin, input.mirrorRange);
}

ObservableScalar evaluate_formula_program(DeviceFormulaProgram program, uint firstInput,
    device const FormulaInstruction* instructions, device const ObservableScalar* constants,
    device const DeviceProgramInput* inputs, device const ObservableScalar* parameters,
    device const ObservableScalar* frame, device ObservableScalar* stack) {
  uint top = 0;
  for( uint i = 0; i < program.instructionCount; ++i ){
    auto instruction = instructions[program.firstInstruction + i];
    auto op = instruction.opcode;
    if( op == FormulaOpcode::LoadInput ){
      stack[top++] = read_program_input(inputs[firstInput + instruction.operand], parameters, frame);
      continue;
    }
    if( op == FormulaOpcode::LoadConstant ){ stack[top++] = constants[program.firstConstant + instruction.operand]; continue; }
    auto value = stack[top - 1];
    switch( op ){
#define GUNDAM_FORMULA_UNARY(opcode, operation) \
      case FormulaOpcode::opcode: value = FormulaMath::operation(value); break;
#include "DeviceMath/SharedFormulaOperations.h"
#undef GUNDAM_FORMULA_UNARY
      default: {
        auto left = stack[top - 2];
        switch( op ){
#define GUNDAM_FORMULA_BINARY(opcode, operation) \
          case FormulaOpcode::opcode: value = FormulaMath::operation(left, value); break;
#include "DeviceMath/SharedFormulaOperations.h"
#undef GUNDAM_FORMULA_BINARY
          default: value = ObservableScalar(NAN); break; // Rejected by host validation.
        }
        --top;
      }
    }
    stack[top - 1] = value;
  }
  return stack[0];
}

ObservableScalar evaluate_program_leaf(DevicePropagationDial dial, ObservableScalar value,
    ObservableScalar second, device const ObservableScalar* payloads,
    bool composite, ObservableScalar inputMin, ObservableScalar inputMax) {
  auto kind = BackendDialType(dial.type);
  device const ObservableScalar* data = payloads + dial.payloadOffset;
  ObservableScalar lower(-INFINITY), upper(INFINITY);
  if( kind == BackendDialType::Bilinear || kind == BackendDialType::Bicubic ){
    return GundamObservableMath::EvaluateSurface(value, second, data,
                kind == BackendDialType::Bicubic, dial.allowExtrapolation != 0);
  }
  if( !dial.allowExtrapolation && kind != BackendDialType::Norm && kind != BackendDialType::Shift ){
    auto first = composite ? inputMin : (kind == BackendDialType::Graph ? data[1] : data[0]);
    auto last = inputMax;
    if( !composite ){
      if( kind == BackendDialType::Graph || kind == BackendDialType::GeneralSpline ){ last = data[dial.payloadSize - 1]; }
      else{
        uint knots = kind == BackendDialType::UniformSpline ? (dial.payloadSize - 2) / 2 : dial.payloadSize - 2;
        last = data[0] + data[1] * ObservableScalar(float(knots - 1));
      }
    }
    // Composite graph nodes use stored endpoint ordinates, not spline bounds.
    if( kind == BackendDialType::Graph ){
      if( value <= data[1] ){ return data[0]; }
      if( value >= data[dial.payloadSize - 1] ){ return data[dial.payloadSize - 2]; }
    }
    else{
      if( value < first ){ value = first; }
      if( value > last ){ value = last; }
    }
  }
  switch( kind ){
    case BackendDialType::Norm: return value;
    case BackendDialType::Shift: return data[0];
    case BackendDialType::CompactSpline: return GundamObservableMath::EvaluateCompactSpline(value, lower, upper, data, dial.payloadSize - 2);
    case BackendDialType::UniformSpline: return GundamObservableMath::EvaluateUniformSpline(value, lower, upper, data, dial.payloadSize);
    case BackendDialType::MonotonicSpline: return GundamObservableMath::EvaluateMonotonicSpline(value, lower, upper, data, dial.payloadSize - 2);
    case BackendDialType::GeneralSpline: return GundamObservableMath::EvaluateGeneralSpline(value, lower, upper, data, dial.payloadSize);
    case BackendDialType::Graph: return GundamObservableMath::EvaluateGraph(value, lower, upper, data, dial.payloadSize);
    default: return ObservableScalar(NAN);
  }
}

ObservableScalar evaluate_composite_program(DevicePropagationDial root, DeviceDialInputRange arguments,
    device const DeviceCompositeProgram* programs, device const DeviceCompositeInstruction* instructions,
    device const DeviceFormulaProgram* formulas, device const FormulaInstruction* formulaInstructions,
    device const ObservableScalar* constants, device const DeviceProgramInput* inputs,
    device const DeviceDialInput* parameterInputs, device const ObservableScalar* parameters,
    device const ObservableScalar* payloads, device ObservableScalar* allScratch) {
  auto program = programs[root.programIndex];
  device ObservableScalar* scratch = allScratch + root.scratchOffset;
  device ObservableScalar* stack = scratch + root.stackOffset;
  for( uint i = 0; i < program.inputCount; ++i ){
    auto input = parameterInputs[arguments.first + i];
    scratch[i] = mirror_program_input(parameters[input.index], input.useMirror, input.mirrorMin, input.mirrorRange);
  }
  for( uint i = 0; i < program.instructionCount; ++i ){
    auto instruction = instructions[program.firstInstruction + i];
    auto dial = instruction.dial;
    ObservableScalar value(0), second(0);
    if( BackendDialType(dial.type) == BackendDialType::Formula ){
      value = evaluate_formula_program(formulas[dial.programIndex], dial.firstProgramInput,
                  formulaInstructions, constants, inputs, parameters, scratch, stack);
    }
    else{
      if( instruction.inputCount ){ value = read_program_input(inputs[instruction.firstInput], parameters, scratch); }
      if( instruction.inputCount > 1 ){ second = read_program_input(inputs[instruction.firstInput + 1], parameters, scratch); }
      value = evaluate_program_leaf(dial, value, second, payloads, true, instruction.inputMin, instruction.inputMax);
    }
    scratch[program.inputCount + i] = Backends::Semantics::processResponse(value, dial);
  }
  return scratch[program.outputSlot];
}
