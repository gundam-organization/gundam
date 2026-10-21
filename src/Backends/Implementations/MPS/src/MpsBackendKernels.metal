constant uint kDialFlagAllowExtrapolation = 1u << 0;
constant uint kDialFlagCached = 1u << 1;

using EventDialRanges = Backends::DeviceEventDialRanges;
using ExternalDialOccurrence = Backends::DeviceExternalDialOccurrence;
using NormDialOccurrence = Backends::DeviceNormDialOccurrence;
using SplineDialDescriptor = Backends::DeviceSplineDialDescriptor;

float evaluate_compact_spline(float x, bool allowExtrapolation, device const float* data, uint dim) {
  if( !allowExtrapolation ){
    float low = data[0];
    float step = data[1];
    float high = low + float(dim - 1) * step;
    x = clamp(x, low, high);
  }
  return GundamDeviceMath::EvaluateCompactSpline(x, -INFINITY, INFINITY, data, int(dim));
}

float evaluate_uniform_spline(float x, bool allowExtrapolation, device const float* data, uint dim) {
  if( !allowExtrapolation ){
    float low = data[0];
    float step = data[1];
    uint knotCount = (dim - 2) / 2;
    float high = low + float(knotCount - 1) * step;
    x = clamp(x, low, high);
  }
  return GundamDeviceMath::EvaluateUniformSpline(x, -INFINITY, INFINITY, data, int(dim));
}

float evaluate_monotonic_spline(float x, bool allowExtrapolation, device const float* data, uint dim) {
  if( !allowExtrapolation ){
    float low = data[0];
    float step = data[1];
    float high = low + float(dim - 1) * step;
    x = clamp(x, low, high);
  }
  return GundamDeviceMath::EvaluateMonotonicSpline(x, -INFINITY, INFINITY, data, int(dim));
}

float evaluate_general_spline(float x, bool allowExtrapolation, device const float* data, uint dim) {
  if( !allowExtrapolation ){
    uint knotCount = (dim - 2) / 3;
    float low = data[0];
    float high = data[2 + 3 * (knotCount - 1) + 2];
    x = clamp(x, low, high);
  }
  return GundamDeviceMath::EvaluateGeneralSpline(x, -INFINITY, INFINITY, data, int(dim));
}

float evaluate_graph(float x, bool allowExtrapolation, device const float* data, uint dim) {
  if( !allowExtrapolation ){
    uint knotCount = dim / 2;
    float low = data[1];
    float high = data[2 * (knotCount - 1) + 1];
    x = clamp(x, low, high);
  }
  return GundamDeviceMath::EvaluateGraph(x, -INFINITY, INFINITY, data, int(dim));
}

// Ordinary in-range responses need only thresholds; transition data is cold.
float process_reduced_response(float value, float lower, float upper, uint smoothingIndex,
                               device const Backends::ResponseSmoothing<float>* smoothings) {
  if( value < lower ){
    if( smoothingIndex == 0 ){ return lower; }
    return Backends::Semantics::processLowerResponse(value, lower, smoothings[smoothingIndex]);
  }
  else if( value > upper ){
    if( smoothingIndex == 0 ){ return upper; }
    return Backends::Semantics::processUpperResponse(value, upper, smoothings[smoothingIndex]);
  }
  return value;
}
template<typename Descriptor>
float process_descriptor_response(float value, Descriptor descriptor,
                                   device const Backends::ResponseSmoothing<float>* smoothings) {
  return Backends::Semantics::processResponse(value, descriptor);
}
float process_descriptor_response(float value, Backends::MpsSplineDialDescriptor descriptor,
                                   device const Backends::ResponseSmoothing<float>* smoothings) {
  return process_reduced_response(value, descriptor.minResponse, descriptor.maxResponse, descriptor.flags >> 2, smoothings);
}
float process_descriptor_response(float value, Backends::MpsNormDialOccurrence descriptor,
                                   device const Backends::ResponseSmoothing<float>* smoothings) {
  return process_reduced_response(value, descriptor.minResponse, descriptor.maxResponse, descriptor.smoothingIndex, smoothings);
}
float process_descriptor_response(float value, Backends::MpsExternalDialOccurrence descriptor,
                                   device const Backends::ResponseSmoothing<float>* smoothings) {
  return process_reduced_response(value, descriptor.minResponse, descriptor.maxResponse, descriptor.smoothingIndex, smoothings);
}

template<typename Descriptor>
float evaluate_compact_descriptor(
    Descriptor descriptor,
    device const float* splineData,
    device const float* parameters,
    device const Backends::ResponseSmoothing<float>* smoothings) {
  float response = evaluate_compact_spline(
      parameters[descriptor.parameterIndex],
      (descriptor.flags & kDialFlagAllowExtrapolation) != 0,
      splineData + descriptor.splineOffset,
      descriptor.splineSize - 2
  );
  return process_descriptor_response(response, descriptor, smoothings);
}

template<typename Descriptor>
float evaluate_uniform_descriptor(
    Descriptor descriptor,
    device const float* splineData,
    device const float* parameters,
    device const Backends::ResponseSmoothing<float>* smoothings) {
  float response = evaluate_uniform_spline(
      parameters[descriptor.parameterIndex],
      (descriptor.flags & kDialFlagAllowExtrapolation) != 0,
      splineData + descriptor.splineOffset,
      descriptor.splineSize
  );
  return process_descriptor_response(response, descriptor, smoothings);
}

template<typename Descriptor>
float evaluate_monotonic_descriptor(
    Descriptor descriptor,
    device const float* splineData,
    device const float* parameters,
    device const Backends::ResponseSmoothing<float>* smoothings) {
  float response = evaluate_monotonic_spline(
      parameters[descriptor.parameterIndex],
      (descriptor.flags & kDialFlagAllowExtrapolation) != 0,
      splineData + descriptor.splineOffset,
      descriptor.splineSize - 2
  );
  return process_descriptor_response(response, descriptor, smoothings);
}

template<typename Descriptor>
float evaluate_general_descriptor(
    Descriptor descriptor,
    device const float* splineData,
    device const float* parameters,
    device const Backends::ResponseSmoothing<float>* smoothings) {
  float response = evaluate_general_spline(
      parameters[descriptor.parameterIndex],
      (descriptor.flags & kDialFlagAllowExtrapolation) != 0,
      splineData + descriptor.splineOffset,
      descriptor.splineSize
  );
  return process_descriptor_response(response, descriptor, smoothings);
}

template<typename Descriptor>
float evaluate_graph_descriptor(
    Descriptor descriptor,
    device const float* splineData,
    device const float* parameters,
    device const Backends::ResponseSmoothing<float>* smoothings) {
  float response = evaluate_graph(
      parameters[descriptor.parameterIndex],
      (descriptor.flags & kDialFlagAllowExtrapolation) != 0,
      splineData + descriptor.splineOffset,
      descriptor.splineSize
  );
  return process_descriptor_response(response, descriptor, smoothings);
}

template<typename NormDescriptor, typename SplineDescriptor, typename ExternalDescriptor>
inline void compute_event_weights_impl(
    device float* eventWeights,
    device const float* baseWeights,
    device const EventDialRanges* eventDialRanges,
    device const NormDescriptor* normDialOccurrences,
    device const float* parameters,
    device const uint* compactDialIndices,
    device const SplineDescriptor* compactDialDescriptors,
    device const float* compactCachedResponses,
    device const uint* uniformDialIndices,
    device const SplineDescriptor* uniformDialDescriptors,
    device const float* uniformCachedResponses,
    device const uint* monotonicDialIndices,
    device const SplineDescriptor* monotonicDialDescriptors,
    device const float* monotonicCachedResponses,
    device const uint* generalDialIndices,
    device const SplineDescriptor* generalDialDescriptors,
    device const float* generalCachedResponses,
    device const uint* graphDialIndices,
    device const SplineDescriptor* graphDialDescriptors,
    device const float* graphCachedResponses,
    device const float* splineData,
    constant uint& nEvents,
    device const float* externalWeights,
    device const ExternalDescriptor* externalDialOccurrences,
    device const Backends::ResponseSmoothing<float>* smoothings,
    uint gid) {
  if( gid >= nEvents ){ return; }

  float weight = baseWeights[gid];
  EventDialRanges ranges = eventDialRanges[gid];

  for( uint iDial = 0 ; iDial < ranges.normCount ; iDial++ ){
    NormDescriptor occurrence = normDialOccurrences[ranges.normOffset + iDial];
    float response = parameters[occurrence.parameterIndex];
    response = process_descriptor_response(response, occurrence, smoothings);
    weight *= response;
  }

  for( uint iDial = 0 ; iDial < ranges.compactCount ; iDial++ ){
    uint packedDial = compactDialIndices[ranges.compactOffset + iDial];
    SplineDescriptor descriptor = compactDialDescriptors[packedDial];
    float response = 1.0f;
    if( (descriptor.flags & kDialFlagCached) != 0 ){
      response = compactCachedResponses[packedDial];
    }
    else{
      response = evaluate_compact_descriptor(descriptor, splineData, parameters, smoothings);
    }
    weight *= response;
  }

  for( uint iDial = 0 ; iDial < ranges.uniformCount ; iDial++ ){
    uint packedDial = uniformDialIndices[ranges.uniformOffset + iDial];
    SplineDescriptor descriptor = uniformDialDescriptors[packedDial];
    float response = 1.0f;
    if( (descriptor.flags & kDialFlagCached) != 0 ){
      response = uniformCachedResponses[packedDial];
    }
    else{
      response = evaluate_uniform_descriptor(descriptor, splineData, parameters, smoothings);
    }
    weight *= response;
  }

  for( uint iDial = 0 ; iDial < ranges.monotonicCount ; iDial++ ){
    uint packedDial = monotonicDialIndices[ranges.monotonicOffset + iDial];
    SplineDescriptor descriptor = monotonicDialDescriptors[packedDial];
    float response = 1.0f;
    if( (descriptor.flags & kDialFlagCached) != 0 ){
      response = monotonicCachedResponses[packedDial];
    }
    else{
      response = evaluate_monotonic_descriptor(descriptor, splineData, parameters, smoothings);
    }
    weight *= response;
  }

  for( uint iDial = 0 ; iDial < ranges.generalCount ; iDial++ ){
    uint packedDial = generalDialIndices[ranges.generalOffset + iDial];
    SplineDescriptor descriptor = generalDialDescriptors[packedDial];
    float response = 1.0f;
    if( (descriptor.flags & kDialFlagCached) != 0 ){
      response = generalCachedResponses[packedDial];
    }
    else{
      response = evaluate_general_descriptor(descriptor, splineData, parameters, smoothings);
    }
    weight *= response;
  }

  for( uint iDial = 0 ; iDial < ranges.graphCount ; iDial++ ){
    uint packedDial = graphDialIndices[ranges.graphOffset + iDial];
    SplineDescriptor descriptor = graphDialDescriptors[packedDial];
    float response = 1.0f;
    if( (descriptor.flags & kDialFlagCached) != 0 ){
      response = graphCachedResponses[packedDial];
    }
    else{
      response = evaluate_graph_descriptor(descriptor, splineData, parameters, smoothings);
    }
    weight *= response;
  }
  for( uint iDial = 0; iDial < ranges.externalCount; ++iDial ){
    ExternalDescriptor occurrence = externalDialOccurrences[ranges.externalOffset + iDial];
    float response = externalWeights[occurrence.weightIndex];
    // Match host comparison semantics, including NaN responses.
    response = process_descriptor_response(response, occurrence, smoothings);
    weight *= response;
  }
  eventWeights[gid] = weight;
}

kernel void compute_cached_compact_responses(
    device float* cachedResponses [[buffer(0)]],
    device const SplineDialDescriptor* descriptors [[buffer(1)]],
    device const float* splineData [[buffer(2)]],
    device const float* parameters [[buffer(3)]],
    constant uint& dispatchCount [[buffer(4)]],
    device const uint* cachedDialIndices [[buffer(5)]],
    constant uint& useCachedIndices [[buffer(6)]],
    device const Backends::ResponseSmoothing<float>* smoothings [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= dispatchCount ){ return; }
  uint descriptorIndex = useCachedIndices ? cachedDialIndices[gid] : gid;
  SplineDialDescriptor descriptor = descriptors[descriptorIndex];
  if( (descriptor.flags & kDialFlagCached) == 0 ){ return; }
  cachedResponses[descriptorIndex] = evaluate_compact_descriptor(descriptor, splineData, parameters, smoothings);
}

kernel void compute_cached_uniform_responses(
    device float* cachedResponses [[buffer(0)]],
    device const SplineDialDescriptor* descriptors [[buffer(1)]],
    device const float* splineData [[buffer(2)]],
    device const float* parameters [[buffer(3)]],
    constant uint& dispatchCount [[buffer(4)]],
    device const uint* cachedDialIndices [[buffer(5)]],
    constant uint& useCachedIndices [[buffer(6)]],
    device const Backends::ResponseSmoothing<float>* smoothings [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= dispatchCount ){ return; }
  uint descriptorIndex = useCachedIndices ? cachedDialIndices[gid] : gid;
  SplineDialDescriptor descriptor = descriptors[descriptorIndex];
  if( (descriptor.flags & kDialFlagCached) == 0 ){ return; }
  cachedResponses[descriptorIndex] = evaluate_uniform_descriptor(descriptor, splineData, parameters, smoothings);
}

kernel void compute_cached_monotonic_responses(
    device float* cachedResponses [[buffer(0)]],
    device const SplineDialDescriptor* descriptors [[buffer(1)]],
    device const float* splineData [[buffer(2)]],
    device const float* parameters [[buffer(3)]],
    constant uint& dispatchCount [[buffer(4)]],
    device const uint* cachedDialIndices [[buffer(5)]],
    constant uint& useCachedIndices [[buffer(6)]],
    device const Backends::ResponseSmoothing<float>* smoothings [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= dispatchCount ){ return; }
  uint descriptorIndex = useCachedIndices ? cachedDialIndices[gid] : gid;
  SplineDialDescriptor descriptor = descriptors[descriptorIndex];
  if( (descriptor.flags & kDialFlagCached) == 0 ){ return; }
  cachedResponses[descriptorIndex] = evaluate_monotonic_descriptor(descriptor, splineData, parameters, smoothings);
}

kernel void compute_cached_general_responses(
    device float* cachedResponses [[buffer(0)]],
    device const SplineDialDescriptor* descriptors [[buffer(1)]],
    device const float* splineData [[buffer(2)]],
    device const float* parameters [[buffer(3)]],
    constant uint& dispatchCount [[buffer(4)]],
    device const uint* cachedDialIndices [[buffer(5)]],
    constant uint& useCachedIndices [[buffer(6)]],
    device const Backends::ResponseSmoothing<float>* smoothings [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= dispatchCount ){ return; }
  uint descriptorIndex = useCachedIndices ? cachedDialIndices[gid] : gid;
  SplineDialDescriptor descriptor = descriptors[descriptorIndex];
  if( (descriptor.flags & kDialFlagCached) == 0 ){ return; }
  cachedResponses[descriptorIndex] = evaluate_general_descriptor(descriptor, splineData, parameters, smoothings);
}

kernel void compute_cached_graph_responses(
    device float* cachedResponses [[buffer(0)]],
    device const SplineDialDescriptor* descriptors [[buffer(1)]],
    device const float* splineData [[buffer(2)]],
    device const float* parameters [[buffer(3)]],
    constant uint& dispatchCount [[buffer(4)]],
    device const uint* cachedDialIndices [[buffer(5)]],
    constant uint& useCachedIndices [[buffer(6)]],
    device const Backends::ResponseSmoothing<float>* smoothings [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= dispatchCount ){ return; }
  uint descriptorIndex = useCachedIndices ? cachedDialIndices[gid] : gid;
  SplineDialDescriptor descriptor = descriptors[descriptorIndex];
  if( (descriptor.flags & kDialFlagCached) == 0 ){ return; }
  cachedResponses[descriptorIndex] = evaluate_graph_descriptor(descriptor, splineData, parameters, smoothings);
}

kernel void compute_event_weights(
    device float* eventWeights [[buffer(0)]],
    device const float* baseWeights [[buffer(1)]],
    device const EventDialRanges* eventDialRanges [[buffer(2)]],
    device const NormDialOccurrence* normDialOccurrences [[buffer(3)]],
    device const float* parameters [[buffer(4)]],
    device const uint* compactDialIndices [[buffer(5)]],
    device const SplineDialDescriptor* compactDialDescriptors [[buffer(6)]],
    device const float* compactCachedResponses [[buffer(7)]],
    device const uint* uniformDialIndices [[buffer(8)]],
    device const SplineDialDescriptor* uniformDialDescriptors [[buffer(9)]],
    device const float* uniformCachedResponses [[buffer(10)]],
    device const uint* monotonicDialIndices [[buffer(11)]],
    device const SplineDialDescriptor* monotonicDialDescriptors [[buffer(12)]],
    device const float* monotonicCachedResponses [[buffer(13)]],
    device const uint* generalDialIndices [[buffer(14)]],
    device const SplineDialDescriptor* generalDialDescriptors [[buffer(15)]],
    device const float* generalCachedResponses [[buffer(16)]],
    device const uint* graphDialIndices [[buffer(17)]],
    device const SplineDialDescriptor* graphDialDescriptors [[buffer(18)]],
    device const float* graphCachedResponses [[buffer(19)]],
    device const float* splineData [[buffer(20)]],
    constant uint& nEvents [[buffer(21)]],
    device const float* externalWeights [[buffer(22)]],
    device const ExternalDialOccurrence* externalDialOccurrences [[buffer(23)]],
    device const Backends::ResponseSmoothing<float>* smoothings [[buffer(24)]],
    uint gid [[thread_position_in_grid]]) {
  compute_event_weights_impl(
      eventWeights, baseWeights, eventDialRanges, normDialOccurrences,
      parameters, compactDialIndices, compactDialDescriptors, compactCachedResponses,
      uniformDialIndices, uniformDialDescriptors, uniformCachedResponses, monotonicDialIndices,
      monotonicDialDescriptors, monotonicCachedResponses, generalDialIndices, generalDialDescriptors,
      generalCachedResponses, graphDialIndices, graphDialDescriptors, graphCachedResponses,
      splineData, nEvents, externalWeights, externalDialOccurrences,
      smoothings, gid);
}

// Compact descriptor entry points share scalar evaluation with the legacy diagnostics.
kernel void compute_cached_compact_responses_reduced(
    device float* cachedResponses [[buffer(0)]],
    device const Backends::MpsSplineDialDescriptor* descriptors [[buffer(1)]],
    device const float* splineData [[buffer(2)]],
    device const float* parameters [[buffer(3)]],
    constant uint& dispatchCount [[buffer(4)]],
    device const uint* cachedDialIndices [[buffer(5)]],
    constant uint& useCachedIndices [[buffer(6)]],
    device const Backends::ResponseSmoothing<float>* smoothings [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= dispatchCount ){ return; }
  uint descriptorIndex = useCachedIndices ? cachedDialIndices[gid] : gid;
  Backends::MpsSplineDialDescriptor descriptor = descriptors[descriptorIndex];
  if( (descriptor.flags & kDialFlagCached) == 0 ){ return; }
  cachedResponses[descriptorIndex] = evaluate_compact_descriptor(descriptor, splineData, parameters, smoothings);
}

kernel void compute_cached_uniform_responses_reduced(
    device float* cachedResponses [[buffer(0)]],
    device const Backends::MpsSplineDialDescriptor* descriptors [[buffer(1)]],
    device const float* splineData [[buffer(2)]],
    device const float* parameters [[buffer(3)]],
    constant uint& dispatchCount [[buffer(4)]],
    device const uint* cachedDialIndices [[buffer(5)]],
    constant uint& useCachedIndices [[buffer(6)]],
    device const Backends::ResponseSmoothing<float>* smoothings [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= dispatchCount ){ return; }
  uint descriptorIndex = useCachedIndices ? cachedDialIndices[gid] : gid;
  Backends::MpsSplineDialDescriptor descriptor = descriptors[descriptorIndex];
  if( (descriptor.flags & kDialFlagCached) == 0 ){ return; }
  cachedResponses[descriptorIndex] = evaluate_uniform_descriptor(descriptor, splineData, parameters, smoothings);
}

kernel void compute_cached_monotonic_responses_reduced(
    device float* cachedResponses [[buffer(0)]],
    device const Backends::MpsSplineDialDescriptor* descriptors [[buffer(1)]],
    device const float* splineData [[buffer(2)]],
    device const float* parameters [[buffer(3)]],
    constant uint& dispatchCount [[buffer(4)]],
    device const uint* cachedDialIndices [[buffer(5)]],
    constant uint& useCachedIndices [[buffer(6)]],
    device const Backends::ResponseSmoothing<float>* smoothings [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= dispatchCount ){ return; }
  uint descriptorIndex = useCachedIndices ? cachedDialIndices[gid] : gid;
  Backends::MpsSplineDialDescriptor descriptor = descriptors[descriptorIndex];
  if( (descriptor.flags & kDialFlagCached) == 0 ){ return; }
  cachedResponses[descriptorIndex] = evaluate_monotonic_descriptor(descriptor, splineData, parameters, smoothings);
}

kernel void compute_cached_general_responses_reduced(
    device float* cachedResponses [[buffer(0)]],
    device const Backends::MpsSplineDialDescriptor* descriptors [[buffer(1)]],
    device const float* splineData [[buffer(2)]],
    device const float* parameters [[buffer(3)]],
    constant uint& dispatchCount [[buffer(4)]],
    device const uint* cachedDialIndices [[buffer(5)]],
    constant uint& useCachedIndices [[buffer(6)]],
    device const Backends::ResponseSmoothing<float>* smoothings [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= dispatchCount ){ return; }
  uint descriptorIndex = useCachedIndices ? cachedDialIndices[gid] : gid;
  Backends::MpsSplineDialDescriptor descriptor = descriptors[descriptorIndex];
  if( (descriptor.flags & kDialFlagCached) == 0 ){ return; }
  cachedResponses[descriptorIndex] = evaluate_general_descriptor(descriptor, splineData, parameters, smoothings);
}

kernel void compute_cached_graph_responses_reduced(
    device float* cachedResponses [[buffer(0)]],
    device const Backends::MpsSplineDialDescriptor* descriptors [[buffer(1)]],
    device const float* splineData [[buffer(2)]],
    device const float* parameters [[buffer(3)]],
    constant uint& dispatchCount [[buffer(4)]],
    device const uint* cachedDialIndices [[buffer(5)]],
    constant uint& useCachedIndices [[buffer(6)]],
    device const Backends::ResponseSmoothing<float>* smoothings [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= dispatchCount ){ return; }
  uint descriptorIndex = useCachedIndices ? cachedDialIndices[gid] : gid;
  Backends::MpsSplineDialDescriptor descriptor = descriptors[descriptorIndex];
  if( (descriptor.flags & kDialFlagCached) == 0 ){ return; }
  cachedResponses[descriptorIndex] = evaluate_graph_descriptor(descriptor, splineData, parameters, smoothings);
}

kernel void compute_event_weights_reduced(
    device float* eventWeights [[buffer(0)]],
    device const float* baseWeights [[buffer(1)]],
    device const EventDialRanges* eventDialRanges [[buffer(2)]],
    device const Backends::MpsNormDialOccurrence* normDialOccurrences [[buffer(3)]],
    device const float* parameters [[buffer(4)]],
    device const uint* compactDialIndices [[buffer(5)]],
    device const Backends::MpsSplineDialDescriptor* compactDialDescriptors [[buffer(6)]],
    device const float* compactCachedResponses [[buffer(7)]],
    device const uint* uniformDialIndices [[buffer(8)]],
    device const Backends::MpsSplineDialDescriptor* uniformDialDescriptors [[buffer(9)]],
    device const float* uniformCachedResponses [[buffer(10)]],
    device const uint* monotonicDialIndices [[buffer(11)]],
    device const Backends::MpsSplineDialDescriptor* monotonicDialDescriptors [[buffer(12)]],
    device const float* monotonicCachedResponses [[buffer(13)]],
    device const uint* generalDialIndices [[buffer(14)]],
    device const Backends::MpsSplineDialDescriptor* generalDialDescriptors [[buffer(15)]],
    device const float* generalCachedResponses [[buffer(16)]],
    device const uint* graphDialIndices [[buffer(17)]],
    device const Backends::MpsSplineDialDescriptor* graphDialDescriptors [[buffer(18)]],
    device const float* graphCachedResponses [[buffer(19)]],
    device const float* splineData [[buffer(20)]],
    constant uint& nEvents [[buffer(21)]],
    device const float* externalWeights [[buffer(22)]],
    device const Backends::MpsExternalDialOccurrence* externalDialOccurrences [[buffer(23)]],
    device const Backends::ResponseSmoothing<float>* smoothings [[buffer(24)]],
    uint gid [[thread_position_in_grid]]) {
  compute_event_weights_impl(
      eventWeights, baseWeights, eventDialRanges, normDialOccurrences,
      parameters, compactDialIndices, compactDialDescriptors, compactCachedResponses,
      uniformDialIndices, uniformDialDescriptors, uniformCachedResponses, monotonicDialIndices,
      monotonicDialDescriptors, monotonicCachedResponses, generalDialIndices, generalDialDescriptors,
      generalCachedResponses, graphDialIndices, graphDialDescriptors, graphCachedResponses,
      splineData, nEvents, externalWeights, externalDialOccurrences,
      smoothings, gid);
}

kernel void fill_histograms(
    device const float* eventWeights [[buffer(0)]],
    device const int* globalBins [[buffer(1)]],
    device float* histSums [[buffer(2)]],
    device float* histSumSquares [[buffer(3)]],
    constant uint& nEvents [[buffer(4)]],
    constant uint& totalBins [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= totalBins ){ return; }

  float sum = 0.0f;
  float sumCompensation = 0.0f;
  float sumSq = 0.0f;
  float sumSqCompensation = 0.0f;
  int bin = int(gid);
  for( uint iEvent = 0 ; iEvent < nEvents ; iEvent++ ){
    if( globalBins[iEvent] != bin ){ continue; }
    float weight = eventWeights[iEvent];
    float correctedWeight = weight - sumCompensation;
    float nextSum = sum + correctedWeight;
    sumCompensation = (nextSum - sum) - correctedWeight;
    sum = nextSum;

    float weightSq = weight * weight;
    float correctedWeightSq = weightSq - sumSqCompensation;
    float nextSumSq = sumSq + correctedWeightSq;
    sumSqCompensation = (nextSumSq - sumSq) - correctedWeightSq;
    sumSq = nextSumSq;
  }

  histSums[gid] = sum;
  histSumSquares[gid] = sumSq;
}

kernel void fill_histogram_partials_by_bin(
    device const float* eventWeights [[buffer(0)]],
    device const uint* binEventOffsets [[buffer(1)]],
    device const uint* binEventIndices [[buffer(2)]],
    device float* partialSums [[buffer(3)]],
    device float* partialSumSquares [[buffer(4)]],
    constant uint& totalBins [[buffer(5)]],
    constant uint& maxChunksPerBin [[buffer(6)]],
    constant uint& chunkSize [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  uint totalPartials = totalBins * maxChunksPerBin;
  if( gid >= totalPartials ){ return; }

  uint bin = gid / maxChunksPerBin;
  uint chunk = gid - bin * maxChunksPerBin;
  uint binBegin = binEventOffsets[bin];
  uint binEnd = binEventOffsets[bin + 1];
  uint begin = binBegin + chunk * chunkSize;
  uint end = min(begin + chunkSize, binEnd);

  float sum = 0.0f;
  float sumCompensation = 0.0f;
  float sumSq = 0.0f;
  float sumSqCompensation = 0.0f;
  for( uint iEntry = begin ; iEntry < end ; iEntry++ ){
    uint iEvent = binEventIndices[iEntry];
    float weight = eventWeights[iEvent];
    float correctedWeight = weight - sumCompensation;
    float nextSum = sum + correctedWeight;
    sumCompensation = (nextSum - sum) - correctedWeight;
    sum = nextSum;

    float weightSq = weight * weight;
    float correctedWeightSq = weightSq - sumSqCompensation;
    float nextSumSq = sumSq + correctedWeightSq;
    sumSqCompensation = (nextSumSq - sumSq) - correctedWeightSq;
    sumSq = nextSumSq;
  }

  partialSums[gid] = sum;
  partialSumSquares[gid] = sumSq;
}

kernel void finalize_histograms_from_partials(
    device const float* partialSums [[buffer(0)]],
    device const float* partialSumSquares [[buffer(1)]],
    device float* histSums [[buffer(2)]],
    device float* histSumSquares [[buffer(3)]],
    constant uint& totalBins [[buffer(4)]],
    constant uint& maxChunksPerBin [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= totalBins ){ return; }

  float sum = 0.0f;
  float sumCompensation = 0.0f;
  float sumSq = 0.0f;
  float sumSqCompensation = 0.0f;
  uint offset = gid * maxChunksPerBin;
  for( uint iChunk = 0 ; iChunk < maxChunksPerBin ; iChunk++ ){
    float partial = partialSums[offset + iChunk];
    float correctedPartial = partial - sumCompensation;
    float nextSum = sum + correctedPartial;
    sumCompensation = (nextSum - sum) - correctedPartial;
    sum = nextSum;

    float partialSq = partialSumSquares[offset + iChunk];
    float correctedPartialSq = partialSq - sumSqCompensation;
    float nextSumSq = sumSq + correctedPartialSq;
    sumSqCompensation = (nextSumSq - sumSq) - correctedPartialSq;
    sumSq = nextSumSq;
  }

  histSums[gid] = sum;
  histSumSquares[gid] = sumSq;
}
