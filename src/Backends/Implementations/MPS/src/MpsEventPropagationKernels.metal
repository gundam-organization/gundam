using namespace Backends;

void propagation_error(device DevicePropagationControl& control, uint error) {
  atomic_fetch_or_explicit((device atomic_uint*)&control.error, error, memory_order_relaxed);
}

kernel void evaluate_propagation_dials(
    device const DevicePropagationDial* dials [[buffer(0)]],
    device const ObservableScalar* parameters [[buffer(1)]],
    device const ObservableScalar* payloads [[buffer(2)]],
    device const ObservableScalar* externalWeights [[buffer(3)]],
    device ObservableScalar* responses [[buffer(4)]],
    constant DevicePropagationControl& control [[buffer(5)]],
    device const uint* dialIndices [[buffer(6)]],
    device const DevicePropagationDialRange* ranges [[buffer(7)]],
    device const DeviceDialInput* dialInputs [[buffer(8)]],
    device const DeviceDialInputRange* inputRanges [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= control.dialCount ){ return; }
  uint dialIndex = gid;
  if( control.dialRangeCount ){
    uint begin = 0, end = control.dialRangeCount;
    while( begin + 1 < end ){
      uint middle = (begin + end) / 2;
      if( gid < ranges[middle].firstThread ){ end = middle; }
      else{ begin = middle; }
    }
    auto range = ranges[begin];
    dialIndex = dialIndices[range.firstDial + gid - range.firstThread];
  }
  auto dial = dials[dialIndex];
  auto kind = BackendDialType(dial.type);
  const bool surface = kind == BackendDialType::Bilinear || kind == BackendDialType::Bicubic;
  ObservableScalar value(0.f);
  if( !surface && kind != BackendDialType::Shift && kind != BackendDialType::ExternalWeight ){
    value = parameters[dial.parameterIndex];
    if( dial.useMirror ){
      value = GundamDeviceMath::remainder(value - dial.mirrorMin, ObservableScalar(2.f) * dial.mirrorRange);
      if( value < ObservableScalar(0.f) ){ value = -value; }
      if( value > dial.mirrorRange ){ value = ObservableScalar(2.f) * dial.mirrorRange - value; }
      value += dial.mirrorMin;
    }
  }
  device const ObservableScalar* data = payloads + dial.payloadOffset;
  if( !surface && !dial.allowExtrapolation && kind != BackendDialType::Norm && kind != BackendDialType::Shift && kind != BackendDialType::ExternalWeight ){
    auto lower = kind == BackendDialType::Graph ? data[1] : data[0];
    ObservableScalar upper;
    if( kind == BackendDialType::Graph || kind == BackendDialType::GeneralSpline ){ upper = data[dial.payloadSize - 1]; }
    else{
      uint knots = kind == BackendDialType::UniformSpline ? (dial.payloadSize - 2) / 2 : dial.payloadSize - 2;
      upper = data[0] + data[1] * ObservableScalar(float(knots - 1));
    }
    if( value < lower ){ value = lower; }
    if( value > upper ){ value = upper; }
  }
  ObservableScalar lower(-INFINITY), upper(INFINITY);
  switch( kind ){
    case BackendDialType::Bilinear:
    case BackendDialType::Bicubic: {
      auto range = inputRanges[dialIndex];
      ObservableScalar values[2];
      for( uint i = 0; i < 2; ++i ){
        auto input = dialInputs[range.first + i];
        auto raw = parameters[input.index];
        if( input.useMirror ){
          raw = GundamDeviceMath::remainder(raw - input.mirrorMin, ObservableScalar(2.f) * input.mirrorRange);
          if( raw < ObservableScalar(0.f) ){ raw = -raw; }
          if( raw > input.mirrorRange ){ raw = ObservableScalar(2.f) * input.mirrorRange - raw; }
          raw += input.mirrorMin;
        }
        values[i] = raw;
      }
      value = GundamObservableMath::EvaluateSurface(values[0], values[1], data,
                  kind == BackendDialType::Bicubic, dial.allowExtrapolation != 0);
      break;
    }
    case BackendDialType::Norm: break;
    case BackendDialType::Shift: value = data[0]; break;
    case BackendDialType::CompactSpline: value = GundamObservableMath::EvaluateCompactSpline(value, lower, upper, data, dial.payloadSize - 2); break;
    case BackendDialType::UniformSpline: value = GundamObservableMath::EvaluateUniformSpline(value, lower, upper, data, dial.payloadSize); break;
    case BackendDialType::MonotonicSpline: value = GundamObservableMath::EvaluateMonotonicSpline(value, lower, upper, data, dial.payloadSize - 2); break;
    case BackendDialType::GeneralSpline: value = GundamObservableMath::EvaluateGeneralSpline(value, lower, upper, data, dial.payloadSize); break;
    case BackendDialType::Graph: value = GundamObservableMath::EvaluateGraph(value, lower, upper, data, dial.payloadSize); break;
    case BackendDialType::ExternalWeight: value = externalWeights[dial.externalIndex]; break;
  }
  responses[dialIndex] = Backends::Semantics::processResponse(value, dial);
}

struct PropagationResponses {
  device const ObservableScalar* values;
  ObservableScalar operator()(uint index) const { return values[index]; }
};

kernel void propagate_event_state(
    device const DevicePropagationEvent* events [[buffer(0)]],
    device const DeviceBinningSample* samples [[buffer(1)]],
    device const DeviceVariableResponse* targets [[buffer(2)]],
    device const uint* weightDialIndices [[buffer(3)]],
    device const uint* variableDialIndices [[buffer(4)]],
    device const ObservableScalar* responses [[buffer(5)]],
    device const ObservableScalar* hostValues [[buffer(6)]],
    device const uint* hostEdits [[buffer(7)]],
    device ObservableScalar* variables [[buffer(8)]],
    device uint* dirty [[buffer(9)]],
    device float* eventWeights [[buffer(10)]],
    constant DevicePropagationControl& control [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= control.eventCount ){ return; }
  auto event = events[gid];
  auto sample = samples[event.sample];
  bool changed = control.firstPropagation || control.forceBinning || sample.readHostVariables;
  if( control.editCount && (sample.readHostVariables || control.forceBinning) ){
    for( uint i = event.firstVariable; i < event.firstVariable + event.variableCount; ++i ){
      if( hostEdits[i] ){
        if( i - event.firstVariable < sample.variableCount && variables[i] != hostValues[i] ){ changed = true; }
        variables[i] = hostValues[i];
      }
    }
  }
  if( control.updateVariables ){
    for( uint i = event.firstResponse; i < event.firstResponse + event.responseCount; ++i ){
      auto target = targets[i];
      auto value = Backends::Semantics::evalVariableResponse(target, variableDialIndices, PropagationResponses{responses});
      if( target.variableIndex - event.firstVariable < sample.variableCount && variables[target.variableIndex] != value ){ changed = true; }
      variables[target.variableIndex] = value;
    }
  }
  dirty[gid] = changed;
  if( control.updateWeights ){
    float weight = 1;
    for( uint i = event.firstDial; i < event.firstDial + event.dialCount; ++i ){ weight *= float(responses[weightDialIndices[i]]); }
    if( control.hasReweightCap && weight > control.maxReweight ){ weight = control.maxReweight; }
    eventWeights[gid] = event.baseWeight * weight;
  }
}

struct RebinOutcome { uint rebinned, candidates, sample, rebuild; };

void store_contribution(device DeviceBinContribution* contributions, uint first, uint slot, uint oldCount,
                        DeviceBinContribution value, thread uint& rebuild) {
  if( !rebuild && (slot >= oldCount || contributions[first + slot].binIndex != value.binIndex) ){ rebuild = 1; }
  contributions[first + slot] = value;
}

void rebin_event(
    device const DevicePropagationEvent* events,
    device const DeviceBinningSample* samples,
    device const DeviceBinningNode* nodes,
    device const DeviceBinningBounds* bounds,
    device const uint* nodeBins,
    device const DeviceBinningBin* bins,
    device const DeviceBinningInterval* intervals,
    device const ObservableScalar* variables,
    device const uint* dirty,
    device int* hardBins,
    device uint* eventCounts,
    device DeviceBinContribution* contributions,
    device DevicePropagationControl& control,
    device const uint* reusableBins,
    uint gid, thread RebinOutcome& outcome) {
  if( gid >= control.eventCount || !dirty[gid] ){ return; }
  auto event = events[gid];
  auto sample = samples[event.sample];
  if( !sample.isDynamic && !sample.isSmoothed ){
    if( !control.firstPropagation ){ return; }
    hardBins[gid] = event.fixedBin;
    eventCounts[gid] = event.fixedBin >= 0 ? 1 : 0;
    if( event.fixedBin >= 0 ){ contributions[event.firstContribution] = {gid, int(sample.binOffset) + event.fixedBin, 1}; }
    outcome.rebuild = 1;
    return;
  }
  for( uint i = 0; i < sample.variableCount; ++i ){
    if( !isfinite(float(variables[event.firstVariable + i])) ){ propagation_error(control, 1); return; }
  }
  outcome.rebinned = 1;
  if( event.sample < control.sampleCount ){ outcome.sample = event.sample; }
  outcome.rebuild = control.firstPropagation;
  uint oldCount = control.firstPropagation ? 0 : eventCounts[gid];
  uint count = 0, firstHardPosition = UINT_MAX, candidateCount = 0, largest = 0;
  // The build certifies that a reusable strict bin cannot hide an earlier bin.
  // HardBins is initialized by the first tree traversal, never by this shortcut.
  if( !control.firstPropagation && !sample.isSmoothed ){
    int previousBin = hardBins[gid];
    if( previousBin >= 0 && uint(previousBin) < sample.binCount ){
      uint position = reusableBins[sample.binOffset + uint(previousBin)];
      if( position != UINT_MAX ){
        auto bin = bins[position];
        ++candidateCount;
        bool contains = true;
        for( uint i = bin.firstInterval; i < bin.firstInterval + bin.intervalCount; ++i ){
          auto interval = intervals[i];
          if( !GundamBinningMath::contains(interval.lowerBound, interval.upperBound,
                                         variables[event.firstVariable + interval.variableIndex]) ){
            contains = false;
            break;
          }
        }
        if( contains ){
          eventCounts[gid] = 1;
          store_contribution(contributions, event.firstContribution, 0, oldCount,
                             {gid, int(sample.binOffset) + previousBin, 1}, outcome.rebuild);
          if( oldCount != 1 ){ outcome.rebuild = 1; }
          outcome.candidates = 1;
          return;
        }
      }
    }
  }
  float total = 0, largestFraction = 0;
  int hardBin = -1;
  uint nodeIndex = sample.firstNode;
  while( nodeIndex < sample.firstNode + sample.nodeCount ){
    auto node = nodes[nodeIndex];
    bool contains = true;
    for( uint axis = 0; axis < sample.variableCount; ++axis ){
      auto bound = bounds[node.firstBounds + axis];
      float value = float(variables[event.firstVariable + axis]);
      if( value < bound.lower || value > bound.upper ){ contains = false; break; }
    }
    if( !contains ){ nodeIndex = node.skip; continue; }
    for( uint j = 0; j < node.binCount; ++j ){
      auto bin = bins[nodeBins[node.firstBin + j]];
      ++candidateCount;
      bool hardContains = true;
      float fraction = 1;
      for( uint i = bin.firstInterval; i < bin.firstInterval + bin.intervalCount; ++i ){
        auto interval = intervals[i];
        auto value = variables[event.firstVariable + interval.variableIndex];
        hardContains = hardContains && GundamBinningMath::contains(interval.lowerBound, interval.upperBound, value);
        if( sample.isSmoothed ){
          fraction *= float(GundamBinningMath::intervalFraction(interval.lowerBound, interval.upperBound,
                          interval.lowerInverseWidth, interval.upperInverseWidth, value));
        }
        // A strict miss rejects the bin immediately. Smoothed support can
        // extend outside the hard bin, so both tests must reject before pruning.
        if( !hardContains && (!sample.isSmoothed || fraction == 0) ){ break; }
      }
      if( hardContains && bin.position < firstHardPosition ){ hardBin = bin.binIndex; firstHardPosition = bin.position; }
      if( !sample.isSmoothed || fraction == 0 ){ continue; }
      if( count + 1 >= event.contributionCapacity ){ propagation_error(control, 4); return; }
      store_contribution(contributions, event.firstContribution, count, oldCount,
                         {gid, int(sample.binOffset) + bin.binIndex, fraction}, outcome.rebuild);
      if( fraction > largestFraction ){ largestFraction = fraction; largest = count; }
      ++count;
      total += fraction;
    }
    ++nodeIndex;
  }
  outcome.candidates = candidateCount;
  hardBins[gid] = hardBin;
  if( !sample.isSmoothed ){
    count = hardBin >= 0 ? 1 : 0;
    if( count ){ store_contribution(contributions, event.firstContribution, 0, oldCount,
                                   {gid, int(sample.binOffset) + hardBin, 1}, outcome.rebuild); }
  }
  else{
    float tolerance = 64 * FLT_EPSILON * float(max(uint(1), count));
    if( !isfinite(total) || total > 1 + tolerance ){ propagation_error(control, 2); return; }
    if( total > 1 ){
      contributions[event.firstContribution + largest].fraction -= total - 1;
      total = 1;
    }
    if( total < 1 ){
      store_contribution(contributions, event.firstContribution, count, oldCount,
                         {gid, int(sample.outsideBin), 1 - total}, outcome.rebuild);
      ++count;
    }
  }
  if( count != oldCount ){ outcome.rebuild = 1; }
  eventCounts[gid] = count;
}

kernel void rebin_events(
    device const DevicePropagationEvent* events [[buffer(0)]],
    device const DeviceBinningSample* samples [[buffer(1)]],
    device const DeviceBinningNode* nodes [[buffer(2)]],
    device const DeviceBinningBounds* bounds [[buffer(3)]],
    device const uint* nodeBins [[buffer(4)]],
    device const DeviceBinningBin* bins [[buffer(5)]],
    device const DeviceBinningInterval* intervals [[buffer(6)]],
    device const ObservableScalar* variables [[buffer(7)]],
    device const uint* dirty [[buffer(8)]],
    device int* hardBins [[buffer(9)]],
    device uint* eventCounts [[buffer(10)]],
    device DeviceBinContribution* contributions [[buffer(11)]],
    device uint* changedSamples [[buffer(12)]],
    device DevicePropagationControl& control [[buffer(13)]],
    device const uint* reusableBins [[buffer(14)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  RebinOutcome outcome{0, 0, UINT_MAX, 0};
  rebin_event(events, samples, nodes, bounds, nodeBins, bins, intervals, variables, dirty,
              hardBins, eventCounts, contributions, control, reusableBins, gid, outcome);
  // All lanes reconverge after the event-local early returns before collectives.
  uint rebinned = outcome.rebinned, candidates = outcome.candidates, rebuild = outcome.rebuild;
  rebinned = simd_sum(rebinned);
  candidates = simd_sum(candidates);
  rebuild = simd_max(rebuild);
  if( lane == 0 ){
    if( rebinned ){ atomic_fetch_add_explicit((device atomic_uint*)&control.rebinnedEvents, rebinned, memory_order_relaxed); }
    if( candidates ){ atomic_fetch_add_explicit((device atomic_uint*)&control.candidates, candidates, memory_order_relaxed); }
    if( rebuild ){ atomic_store_explicit((device atomic_uint*)&control.rebuildIndex, 1, memory_order_relaxed); }
  }
  uint firstSample = simd_broadcast_first(outcome.sample);
  if( outcome.sample != UINT_MAX && (lane == 0 || outcome.sample != firstSample) ){
    atomic_store_explicit((device atomic_uint*)(changedSamples + outcome.sample), 1, memory_order_relaxed);
  }
}

kernel void clear_contribution_counts(
    device atomic_uint* counts [[buffer(0)]], device const DevicePropagationControl& control [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid < control.histogramBins && control.rebuildIndex ){ atomic_store_explicit(counts + gid, 0, memory_order_relaxed); }
}

kernel void count_bin_contributions(
    device const DevicePropagationEvent* events [[buffer(0)]], device const uint* eventCounts [[buffer(1)]],
    device const DeviceBinContribution* contributions [[buffer(2)]], device atomic_uint* counts [[buffer(3)]],
    device const DevicePropagationControl& control [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if( gid >= control.eventCount || !control.rebuildIndex || control.error ){ return; }
  auto event = events[gid];
  for( uint i = 0; i < eventCounts[gid]; ++i ){ atomic_fetch_add_explicit(counts + contributions[event.firstContribution + i].binIndex, 1, memory_order_relaxed); }
}

kernel void index_histogram_chunks(
    device const uint* counts [[buffer(0)]], device uint* offsets [[buffer(1)]],
    device uint* positions [[buffer(2)]], device uint* chunkOffsets [[buffer(3)]],
    device DeviceHistogramChunk* chunks [[buffer(4)]], device DevicePropagationControl& control [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid || !control.rebuildIndex || control.error ){ return; }
  uint offset = 0, chunk = 0;
  for( uint bin = 0; bin < control.histogramBins; ++bin ){
    offsets[bin] = positions[bin] = offset;
    chunkOffsets[bin] = chunk;
    uint end = offset + counts[bin];
    while( offset < end ){
      uint next = min(offset + 256, end);
      chunks[chunk++] = {bin, offset, next};
      offset = next;
    }
  }
  offsets[control.histogramBins] = offset;
  chunkOffsets[control.histogramBins] = chunk;
  control.chunkCount = chunk;
  control.contributionCount = offset;
}

kernel void scatter_bin_contributions(
    device const DevicePropagationEvent* events [[buffer(0)]], device const uint* eventCounts [[buffer(1)]],
    device const DeviceBinContribution* contributions [[buffer(2)]], device atomic_uint* positions [[buffer(3)]],
    device uint* binContributions [[buffer(4)]], device const DevicePropagationControl& control [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= control.eventCount || !control.rebuildIndex || control.error ){ return; }
  auto event = events[gid];
  for( uint i = 0; i < eventCounts[gid]; ++i ){
    uint index = event.firstContribution + i;
    uint position = atomic_fetch_add_explicit(positions + contributions[index].binIndex, 1, memory_order_relaxed);
    binContributions[position] = index;
  }
}

void compensated_add(float value, thread float& sum, thread float& compensation) {
  float corrected = value - compensation;
  float next = sum + corrected;
  compensation = (next - sum) - corrected;
  sum = next;
}

kernel void reduce_histogram_chunks(
    device const DeviceHistogramChunk* chunks [[buffer(0)]], device const uint* indices [[buffer(1)]],
    device const DeviceBinContribution* contributions [[buffer(2)]], device const float* weights [[buffer(3)]],
    device float* sums [[buffer(4)]], device float* squares [[buffer(5)]],
    device const DevicePropagationControl& control [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  if( gid >= control.chunkCount || control.error ){ return; }
  auto chunk = chunks[gid];
  float sum = 0, compensation = 0, square = 0, squareCompensation = 0;
  for( uint i = chunk.begin; i < chunk.end; ++i ){
    auto contribution = contributions[indices[i]];
    float value = weights[contribution.event] * contribution.fraction;
    compensated_add(value, sum, compensation);
    compensated_add(value * value, square, squareCompensation);
  }
  sums[gid] = sum;
  squares[gid] = square;
}

kernel void finish_dynamic_histograms(
    device const uint* chunkOffsets [[buffer(0)]], device const float* partialSums [[buffer(1)]],
    device const float* partialSquares [[buffer(2)]], device float* sums [[buffer(3)]], device float* squares [[buffer(4)]],
    device const DevicePropagationControl& control [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  if( gid >= control.histogramBins || control.error ){ return; }
  float sum = 0, compensation = 0, square = 0, squareCompensation = 0;
  for( uint i = chunkOffsets[gid]; i < chunkOffsets[gid + 1]; ++i ){
    compensated_add(partialSums[i], sum, compensation);
    compensated_add(partialSquares[i], square, squareCompensation);
  }
  sums[gid] = sum;
  squares[gid] = square;
}

kernel void reduce_histogram_chunks_parallel(
    device const DeviceHistogramChunk* chunks [[buffer(0)]], device const uint* indices [[buffer(1)]],
    device const DeviceBinContribution* contributions [[buffer(2)]], device const float* weights [[buffer(3)]],
    device float* sums [[buffer(4)]], device float* squares [[buffer(5)]],
    device const DevicePropagationControl& control [[buffer(6)]], uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint chunkIndex = gid / 32;
  if( chunkIndex >= control.chunkCount || control.error ){ return; }
  auto chunk = chunks[chunkIndex];
  ObservableScalar sum(0.f), square(0.f);
  for( uint i = chunk.begin + lane; i < chunk.end; i += 32 ){
    auto contribution = contributions[indices[i]];
    float value = weights[contribution.event] * contribution.fraction;
    sum += ObservableScalar(value);
    square += ObservableScalar(value * value);
  }
  for( uint stride = 16; stride; stride /= 2 ){
    ObservableScalar otherSum, otherSquare;
    otherSum.high = simd_shuffle_down(sum.high, stride);
    otherSum.low = simd_shuffle_down(sum.low, stride);
    otherSquare.high = simd_shuffle_down(square.high, stride);
    otherSquare.low = simd_shuffle_down(square.low, stride);
    if( lane + stride < 32 ){ sum += otherSum; square += otherSquare; }
  }
  if( lane == 0 ){ sums[chunkIndex] = float(sum); squares[chunkIndex] = float(square); }
}
