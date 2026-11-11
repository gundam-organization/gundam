kernel void gradient_dials(
    device const Backends::MpsGradientInput* directions [[buffer(0)]],
    device const Backends::DevicePropagationDial* dials [[buffer(1)]],
    device const Backends::DeviceDialInput* dialInputs [[buffer(2)]],
    device const Backends::DeviceDialInputRange* ranges [[buffer(3)]],
    device const Backends::ObservableScalar* parameters [[buffer(4)]],
    device const Backends::ObservableScalar* payloads [[buffer(5)]],
    device const Backends::ObservableScalar* externalWeights [[buffer(6)]],
    device const Backends::ObservableScalar* externalSlopes [[buffer(7)]],
    device const Backends::DeviceFormulaProgram* formulas [[buffer(8)]],
    device const Backends::FormulaInstruction* formulaInstructions [[buffer(9)]],
    device const Backends::ObservableScalar* constants [[buffer(10)]],
    device const Backends::DeviceProgramInput* programInputs [[buffer(11)]],
    device const Backends::DeviceCompositeProgram* programs [[buffer(12)]],
    device const Backends::DeviceCompositeInstruction* instructions [[buffer(13)]],
    device MpsGradient::Dual* allScratch [[buffer(14)]],
    device Backends::ObservableScalar* out [[buffer(15)]],
    constant Backends::GradientWork<Backends::ObservableScalar>& work [[buffer(16)]], uint i [[thread_position_in_grid]]) {
  using namespace Backends;
  using namespace MpsGradient;
  if( i >= work.dials ) return;
  auto direction = directions[i];
  auto dial = dials[direction.dial];
  auto range = ranges[direction.dial];
  auto kind = BackendDialType(dial.type);
  device Dual* scratch = allScratch+direction.scratch;
  Dual value;
  if( kind == BackendDialType::ExternalWeight || kind == BackendDialType::Tabulated || kind == BackendDialType::Kriged ){
    auto input = dialInputs[range.first+direction.input];
    auto transformed = mirror(Dual(parameters[input.index], S(1)), input.useMirror, input.mirrorMin, input.mirrorRange);
    S raw(0);
    if( kind == BackendDialType::ExternalWeight ) raw = externalWeights[dial.externalIndex];
    else for( uint j = 0; j < dial.payloadSize; j += 2 )
      raw += payloads[dial.payloadOffset+j+1]*externalWeights[dial.externalIndex+table_term_index(payloads[dial.payloadOffset+j])];
    value = Dual(raw, chain(transformed.slope, externalSlopes[direction.external]));
  }
  else if( kind == BackendDialType::Composite ){
    auto program = programs[dial.programIndex];
    for( uint j = 0; j < range.count; ++j ){
      auto input = dialInputs[range.first+j];
      scratch[j] = mirror(Dual(parameters[input.index], j == direction.input ? S(1) : S(0)), input.useMirror, input.mirrorMin, input.mirrorRange);
    }
    for( uint j = 0; j < program.instructionCount; ++j ){
      auto instruction = instructions[program.firstInstruction+j];
      auto node = instruction.dial;
      Dual result;
      if( BackendDialType(node.type) == BackendDialType::Formula ){
        result = formula(formulas[node.programIndex], node.firstProgramInput, formulaInstructions,
            constants, programInputs, parameters, scratch, scratch+dial.stackOffset, uint(-1));
      }
      else{
        Dual first, second;
        if( instruction.inputCount ) first = read(programInputs[instruction.firstInput], parameters, scratch, uint(-1));
        if( instruction.inputCount > 1 ) second = read(programInputs[instruction.firstInput+1], parameters, scratch, uint(-1));
        result = leaf(node, first, second, payloads, true, instruction.inputMin, instruction.inputMax);
      }
      scratch[program.inputCount+j] = supervise(result, node);
    }
    value = scratch[program.outputSlot];
  }
  else if( kind == BackendDialType::Formula ){
    uint parameter = dialInputs[range.first+direction.input].index;
    // Formula program bindings have resolved parameter indices. Differentiate
    // all uses once, assigning the result to the first binding of that parameter.
    for( uint j = 0; j < direction.input; ++j ){
      if( dialInputs[range.first+j].index == parameter ){ out[i] = S(0); return; }
    }
    value = formula(formulas[dial.programIndex], dial.firstProgramInput, formulaInstructions,
        constants, programInputs, parameters, scratch, scratch, parameter);
  }
  else{
    Dual first, second;
    if( range.count ){
      auto input = dialInputs[range.first];
      first = mirror(Dual(parameters[input.index], direction.input == 0 ? S(1) : S(0)), input.useMirror, input.mirrorMin, input.mirrorRange);
    }
    if( range.count > 1 ){
      auto input = dialInputs[range.first+1];
      second = mirror(Dual(parameters[input.index], direction.input == 1 ? S(1) : S(0)), input.useMirror, input.mirrorMin, input.mirrorRange);
    }
    value = leaf(dial, first, second, payloads, false, S(0), S(0));
  }
  out[i] = supervise(value, dial).slope;
}
kernel void gradient_bins(
    device const Backends::LikelihoodBin<Backends::ObservableScalar>* bins [[buffer(0)]],
    device const Backends::ObservableScalar* sums [[buffer(1)]],
    device const Backends::ObservableScalar* squares [[buffer(2)]],
    device Backends::GradientSemantics::DialEvaluation<Backends::ObservableScalar>* out [[buffer(3)]],
    constant Backends::GradientWork<Backends::ObservableScalar>& work [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if( i < work.bins ) out[i] = Backends::GradientSemantics::binScales<Backends::LikelihoodSemantics::MetalMath>(bins[i], sums[i], squares[i]);
}
kernel void gradient_events(
    device const Backends::MpsGradientEvent* events [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device const uint* inputOffsets [[buffer(2)]],
    device const Backends::ObservableScalar* responses [[buffer(3)]],
    device const Backends::ObservableScalar* slopes [[buffer(4)]],
    device const Backends::GradientSemantics::DialEvaluation<Backends::ObservableScalar>* scales [[buffer(5)]],
    device const Backends::ObservableScalar* weights [[buffer(6)]],
    device Backends::ObservableScalar* out [[buffer(7)]],
    constant Backends::GradientWork<Backends::ObservableScalar>& work [[buffer(8)]], uint i [[thread_position_in_grid]]) {
  using S = Backends::ObservableScalar;
  if( i >= work.events ) return;
  auto event = events[i];
  S prefix(1);
  uint offset = event.firstContribution;
  for( uint j = 0; j < event.count; ++j ){
    uint dial = indices[event.first+j];
    for( uint k = inputOffsets[dial]; k < inputOffsets[dial+1]; ++k ) out[offset++] = prefix;
    prefix *= responses[dial];
  }
  S scale(0);
  if( event.bin < work.bins && !(work.useCap && prefix > work.cap) ){
    auto bin = scales[event.bin];
    scale = event.base*(bin.value+weights[event.result]*bin.derivative);
  }
  S suffix(1);
  for( uint j = event.count; j-- > 0; ){
    uint dial = indices[event.first+j];
    for( uint k = inputOffsets[dial+1]; k-- > inputOffsets[dial]; ){
      --offset;
      out[offset] = scale == S(0) ? S(0) : scale*out[offset]*suffix*slopes[k];
    }
    suffix *= responses[dial];
  }
}
kernel void gradient_reduce(
    device const uint* offsets [[buffer(0)]], device const uint* occurrences [[buffer(1)]],
    device const Backends::ObservableScalar* contributions [[buffer(2)]],
    device Backends::ObservableScalar* out [[buffer(3)]],
    uint parameter [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  using Backends::ObservableScalar;
  threadgroup ObservableScalar values[128];
  ObservableScalar value(0);
  for( uint i = offsets[parameter]+lane; i < offsets[parameter+1]; i += 128 ) value += contributions[occurrences[i]];
  values[lane] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for( uint stride = 64; stride; stride /= 2 ){
    if( lane < stride ) values[lane] += values[lane+stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if( lane == 0 ) out[parameter] = values[0];
}
