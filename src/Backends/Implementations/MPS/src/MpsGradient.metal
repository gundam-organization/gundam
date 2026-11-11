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
namespace MpsGradient {
S kernelSlope(S x) {
  if( x <= S(-1) || x >= S(1) ) return S(0);
  S t = S(.5f)*(x+S(1));
  return S(15)*t*t*(S(1)-t)*(S(1)-t);
}
// Differentiate the same BVH candidates and compact-support products as value
// propagation. Strict selection intervals are constant. No event/parameter or
// event/bin dense Jacobian is allocated.
S observableScale(DevicePropagationEvent event, DeviceBinningSample sample, uint column, S weight,
    device const DeviceBinningNode* nodes, device const DeviceBinningBounds* bounds,
    device const uint* nodeBins, device const DeviceBinningBin* bins,
    device const DeviceBinningInterval* intervals, device const S* variables,
    device const GradientSemantics::DialEvaluation<S>* scales) {
  if( !sample.isSmoothed || column >= sample.variableCount ) return S(0);
  S result(0), total(0), totalSlope(0), largestFraction(0), largestSlope(0);
  GradientSemantics::DialEvaluation<S> largestScales{};
  uint nodeIndex = sample.firstNode;
  while( nodeIndex < sample.firstNode+sample.nodeCount ){
    auto node = nodes[nodeIndex];
    bool contains = true;
    for( uint axis = 0; axis < sample.variableCount; ++axis ){
      auto bound = bounds[node.firstBounds+axis];
      float value = float(variables[event.firstVariable+axis]);
      if( value < bound.lower || value > bound.upper ){ contains = false; break; }
    }
    if( !contains ){ nodeIndex = node.skip; continue; }
    for( uint j = 0; j < node.binCount; ++j ){
      auto bin = bins[nodeBins[node.firstBin+j]];
      Dual fraction(S(1));
      for( uint k = 0; k < bin.intervalCount; ++k ){
        auto interval = intervals[bin.firstInterval+k];
        S value = variables[event.firstVariable+interval.variableIndex];
        S f = GundamBinningMath::intervalFraction(interval.lowerBound, interval.upperBound,
            interval.lowerInverseWidth, interval.upperInverseWidth, value);
        S df(0);
        if( f > S(0) && interval.variableIndex == column && interval.lowerInverseWidth > S(0) ){
          df = kernelSlope((interval.lowerBound-value)*interval.lowerInverseWidth)*interval.lowerInverseWidth
             - kernelSlope((interval.upperBound-value)*interval.upperInverseWidth)*interval.upperInverseWidth;
        }
        fraction = fraction*Dual(f,df);
        if( fraction.value == S(0) ) break;
      }
      if( fraction.value == S(0) ) continue;
      auto scale = scales[sample.binOffset+uint(bin.binIndex)];
      result += weight*(scale.value+weight*fraction.value*scale.derivative)*fraction.slope;
      total += fraction.value;
      totalSlope += fraction.slope;
      if( fraction.value > largestFraction ){
        largestFraction = fraction.value; largestSlope = fraction.slope; largestScales = scale;
      }
    }
    ++nodeIndex;
  }
  // Propagation removes small numerical excesses from the largest row. Its
  // derivative loses the total slope; outside fractions never enter the LLH.
  if( total > S(1) ){
    S corrected = largestFraction-(total-S(1));
    result += weight*((largestScales.value+weight*corrected*largestScales.derivative)*(largestSlope-totalSlope)
              -(largestScales.value+weight*largestFraction*largestScales.derivative)*largestSlope);
  }
  return result;
}
void writeDial(S scale, uint dial, thread uint& offset, device const uint* inputOffsets,
               device const S* slopes, device S* out) {
  for( uint k = inputOffsets[dial]; k < inputOffsets[dial+1]; ++k )
    out[offset++] = scale == S(0) ? S(0) : scale*slopes[k];
}
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
    constant Backends::GradientWork<Backends::ObservableScalar>& work [[buffer(8)]],
    device const Backends::DevicePropagationEvent* propagationEvents [[buffer(9)]],
    device const Backends::DeviceBinningSample* samples [[buffer(10)]],
    device const Backends::DeviceVariableResponse* targets [[buffer(11)]],
    device const uint* variableDials [[buffer(12)]],
    device const Backends::ObservableScalar* variables [[buffer(13)]],
    device const uint* counts [[buffer(14)]],
    device const Backends::DeviceBinContribution* contributions [[buffer(15)]],
    device const Backends::DeviceBinningNode* nodes [[buffer(16)]],
    device const Backends::DeviceBinningBounds* bounds [[buffer(17)]],
    device const uint* nodeBins [[buffer(18)]],
    device const Backends::DeviceBinningBin* bins [[buffer(19)]],
    device const Backends::DeviceBinningInterval* intervals [[buffer(20)]], uint i [[thread_position_in_grid]]) {
  using namespace Backends;
  using namespace MpsGradient;
  if( i >= work.events ) return;
  auto event = events[i];
  auto propagated = propagationEvents[event.result];
  S weight = weights[event.result];
  S prefix(1);
  uint offset = event.firstContribution;
  for( uint j = 0; j < event.count; ++j ){
    uint dial = indices[event.first+j];
    for( uint k = inputOffsets[dial]; k < inputOffsets[dial+1]; ++k ) out[offset++] = prefix;
    prefix *= responses[dial];
  }
  uint observableOffset = offset;
  S scale(0);
  if( !(work.useCap && prefix > work.cap) ){
    for( uint j = 0; j < counts[event.result]; ++j ){
      auto row = contributions[propagated.firstContribution+j];
      if( row.binIndex < 0 || uint(row.binIndex) >= work.bins ) continue;
      auto bin = scales[uint(row.binIndex)];
      scale += row.fraction*(bin.value+weight*row.fraction*bin.derivative);
    }
    scale *= event.base;
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
  // A global weight cap suppresses only weight derivatives, not migration.
  offset = observableOffset;
  auto sample = samples[propagated.sample];
  for( uint j = 0; j < propagated.responseCount; ++j ){
    auto target = targets[propagated.firstResponse+j];
    S adjoint = observableScale(propagated, sample, target.variableIndex-propagated.firstVariable,
        weight, nodes, bounds, nodeBins, bins, intervals, variables, scales);
    bool hasOverride = target.overrideDial != UINT_MAX;
    uint overrideOffset = offset;
    if( hasOverride ) offset += inputOffsets[target.overrideDial+1]-inputOffsets[target.overrideDial];
    S product = hasOverride ? responses[target.overrideDial] : target.referenceValue;
    // Store prefixes in the output slots, then complete them backwards. This
    // preserves repeated and zero multipliers without dividing by responses.
    for( uint k = 0; k < target.multiplyDialCount; ++k ){
      uint dial = variableDials[target.firstMultiplyDial+k];
      for( uint d = inputOffsets[dial]; d < inputOffsets[dial+1]; ++d ) out[offset++] = product;
      product *= responses[dial];
    }
    uint afterMultiply = offset;
    S trailing(1);
    for( uint k = target.multiplyDialCount; k-- > 0; ){
      uint dial = variableDials[target.firstMultiplyDial+k];
      for( uint d = inputOffsets[dial+1]; d-- > inputOffsets[dial]; ){
        --offset;
        out[offset] = adjoint == S(0) ? S(0) : adjoint*out[offset]*trailing*slopes[d];
      }
      trailing *= responses[dial];
    }
    if( hasOverride ) writeDial(adjoint*trailing, target.overrideDial, overrideOffset, inputOffsets, slopes, out);
    offset = afterMultiply;
    for( uint k = 0; k < target.addDialCount; ++k )
      writeDial(adjoint, variableDials[target.firstAddDial+k], offset, inputOffsets, slopes, out);
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
