constant uint kDialTypeNorm = 0;
constant uint kDialTypeCompactSpline = 1;
constant uint kDialTypeUniformSpline = 2;
constant uint kDialTypeMonotonicSpline = 3;
constant uint kDialTypeGeneralSpline = 4;
constant uint kDialTypeGraph = 5;

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
    float low = data[2];
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

kernel void compute_event_weights(
    device float* eventWeights [[buffer(0)]],
    device const float* baseWeights [[buffer(1)]],
    device const uint* eventDialOffsets [[buffer(2)]],
    device const uint* eventDialCounts [[buffer(3)]],
    device const uint* eventDialIndices [[buffer(4)]],
    device const uint* dialTypes [[buffer(5)]],
    device const uint* dialParameterIndices [[buffer(6)]],
    device const float* dialMinResponses [[buffer(7)]],
    device const float* dialMaxResponses [[buffer(8)]],
    device const uint* dialSplineOffsets [[buffer(9)]],
    device const uint* dialSplineSizes [[buffer(10)]],
    device const uint* dialAllowExtrapolation [[buffer(11)]],
    device const float* splineData [[buffer(12)]],
    device const float* parameters [[buffer(13)]],
    constant uint& nEvents [[buffer(14)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= nEvents ){ return; }

  float weight = baseWeights[gid];
  uint dialOffset = eventDialOffsets[gid];
  uint dialCount = eventDialCounts[gid];
  for( uint iDial = 0 ; iDial < dialCount ; iDial++ ){
    uint packedDial = eventDialIndices[dialOffset + iDial];
    float input = parameters[dialParameterIndices[packedDial]];
    float response = 1.0f;
    uint dialType = dialTypes[packedDial];
    if( dialType == kDialTypeNorm ){
      response = input;
    }
    else if( dialType == kDialTypeCompactSpline ){
      uint splineOffset = dialSplineOffsets[packedDial];
      uint splineSize = dialSplineSizes[packedDial];
      response = evaluate_compact_spline(
          input,
          dialAllowExtrapolation[packedDial] != 0,
          splineData + splineOffset,
          splineSize - 2
      );
    }
    else if( dialType == kDialTypeUniformSpline ){
      uint splineOffset = dialSplineOffsets[packedDial];
      uint splineSize = dialSplineSizes[packedDial];
      response = evaluate_uniform_spline(
          input,
          dialAllowExtrapolation[packedDial] != 0,
          splineData + splineOffset,
          splineSize
      );
    }
    else if( dialType == kDialTypeMonotonicSpline ){
      uint splineOffset = dialSplineOffsets[packedDial];
      uint splineSize = dialSplineSizes[packedDial];
      response = evaluate_monotonic_spline(
          input,
          dialAllowExtrapolation[packedDial] != 0,
          splineData + splineOffset,
          splineSize - 2
      );
    }
    else if( dialType == kDialTypeGeneralSpline ){
      uint splineOffset = dialSplineOffsets[packedDial];
      uint splineSize = dialSplineSizes[packedDial];
      response = evaluate_general_spline(
          input,
          dialAllowExtrapolation[packedDial] != 0,
          splineData + splineOffset,
          splineSize
      );
    }
    else if( dialType == kDialTypeGraph ){
      uint splineOffset = dialSplineOffsets[packedDial];
      uint splineSize = dialSplineSizes[packedDial];
      response = evaluate_graph(
          input,
          dialAllowExtrapolation[packedDial] != 0,
          splineData + splineOffset,
          splineSize
      );
    }
    response = max(response, dialMinResponses[packedDial]);
    response = min(response, dialMaxResponses[packedDial]);
    weight *= response;
  }
  eventWeights[gid] = weight;
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
