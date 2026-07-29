#include <metal_stdlib>
using namespace metal;

constant uint kDialTypeNorm = 0;
constant uint kDialTypeCompactSpline = 1;
constant uint kDialTypeUniformSpline = 2;
constant uint kDialTypeMonotonicSpline = 3;

float evaluate_compact_spline(float x, bool allowExtrapolation, device const float* data, uint dim) {
  float low = data[0];
  float step = data[1];
  if( !allowExtrapolation ){
    float high = low + float(dim - 1) * step;
    x = clamp(x, low, high);
  }

  float xx = (x - low) / step;
  int ix = int(floor(xx));

  int d21_0 = ix - 1;
  if( d21_0 < 0 ){ d21_0 = 0; }
  if( d21_0 > int(dim) - 2 ){ d21_0 = int(dim) - 2; }
  int d21_1 = d21_0 + 1;

  int d32_0 = ix;
  if( d32_0 < 0 ){ d32_0 = 0; }
  if( d32_0 > int(dim) - 2 ){ d32_0 = int(dim) - 2; }
  int d32_1 = d32_0 + 1;

  int d43_0 = ix + 1;
  if( d43_0 < 0 ){ d43_0 = 0; }
  if( d43_0 > int(dim) - 2 ){ d43_0 = int(dim) - 2; }
  int d43_1 = d43_0 + 1;

  float p2 = data[2 + d32_0];
  float p3 = data[2 + d32_1];
  float fx = xx - float(d32_0);
  float d21 = data[2 + d21_1] - data[2 + d21_0];
  float d32 = p3 - p2;
  float d43 = data[2 + d43_1] - data[2 + d43_0];
  float m2 = 0.5f * (d21 + d32);
  float m3 = 0.5f * (d32 + d43);

  return ((((2.0f * p2 - 2.0f * p3 + m3 + m2) * fx
            + 3.0f * p3 - 3.0f * p2 - m3 - 2.0f * m2) * fx
           + m2) * fx
          + p2);
}

float evaluate_uniform_spline(float x, bool allowExtrapolation, device const float* data, uint dim) {
  float low = data[0];
  float step = data[1];
  uint knotCount = (dim - 2) / 2;
  if( !allowExtrapolation ){
    float high = low + float(knotCount - 1) * step;
    x = clamp(x, low, high);
  }

  float xx = (x - low) / step;
  int ix = int(xx);
  if( ix < 0 ){ ix = 0; }
  if( 2 * ix + 7 > int(dim) ){ ix = int((dim - 2) / 2) - 2; }

  float fx = xx - float(ix);
  float p1 = data[2 + 2 * ix];
  float m1 = data[2 + 2 * ix + 1] * step;
  float p2 = data[2 + 2 * ix + 2];
  float m2 = data[2 + 2 * ix + 3] * step;

  return ((((2.0f * p1 - 2.0f * p2 + m2 + m1) * fx
            + 3.0f * p2 - 3.0f * p1 - m2 - 2.0f * m1) * fx
           + m1) * fx
          + p1);
}

float evaluate_monotonic_spline(float x, bool allowExtrapolation, device const float* data, uint dim) {
  float low = data[0];
  float step = data[1];
  if( !allowExtrapolation ){
    float high = low + float(dim - 1) * step;
    x = clamp(x, low, high);
  }

  float xx = (x - low) / step;
  int ix = int(floor(xx));

  int d21_0 = ix - 1;
  if( d21_0 < 0 ){ d21_0 = 0; }
  if( d21_0 > int(dim) - 2 ){ d21_0 = int(dim) - 2; }
  int d21_1 = d21_0 + 1;

  int d32_0 = ix;
  if( d32_0 < 0 ){ d32_0 = 0; }
  if( d32_0 > int(dim) - 2 ){ d32_0 = int(dim) - 2; }
  int d32_1 = d32_0 + 1;

  int d43_0 = ix + 1;
  if( d43_0 < 0 ){ d43_0 = 0; }
  if( d43_0 > int(dim) - 2 ){ d43_0 = int(dim) - 2; }
  int d43_1 = d43_0 + 1;

  float p2 = data[2 + d32_0];
  float p3 = data[2 + d32_1];
  float fx = xx - float(d32_0);
  float d21 = data[2 + d21_1] - data[2 + d21_0];
  float d32 = p3 - p2;
  float d43 = data[2 + d43_1] - data[2 + d43_0];
  float m2 = 0.5f * (d21 + d32);
  float m3 = 0.5f * (d32 + d43);

  if( d32 * d21 <= 0.0f ){ m2 = 0.0f; }
  if( d43 * d32 <= 0.0f ){ m3 = 0.0f; }

  float delta2 = 3.0f * min(abs(d21), abs(d32));
  float delta3 = 3.0f * min(abs(d32), abs(d43));
  m2 = clamp(m2, -delta2, delta2);
  m3 = clamp(m3, -delta3, delta3);

  return ((((2.0f * p2 - 2.0f * p3 + m3 + m2) * fx
            + 3.0f * p3 - 3.0f * p2 - m3 - 2.0f * m2) * fx
           + m2) * fx
          + p2);
}

kernel void compute_event_weights(
    device float* eventWeights [[buffer(0)]],
    device const float* baseWeights [[buffer(1)]],
    device const uint* eventDialOffsets [[buffer(2)]],
    device const uint* eventDialCounts [[buffer(3)]],
    device const uint* dialTypes [[buffer(4)]],
    device const uint* dialParameterIndices [[buffer(5)]],
    device const float* dialMinResponses [[buffer(6)]],
    device const float* dialMaxResponses [[buffer(7)]],
    device const uint* dialSplineOffsets [[buffer(8)]],
    device const uint* dialSplineSizes [[buffer(9)]],
    device const uint* dialAllowExtrapolation [[buffer(10)]],
    device const float* splineData [[buffer(11)]],
    device const float* parameters [[buffer(12)]],
    constant uint& nEvents [[buffer(13)]],
    uint gid [[thread_position_in_grid]]) {
  if( gid >= nEvents ){ return; }

  float weight = baseWeights[gid];
  uint dialOffset = eventDialOffsets[gid];
  uint dialCount = eventDialCounts[gid];
  for( uint iDial = 0 ; iDial < dialCount ; iDial++ ){
    uint flatDial = dialOffset + iDial;
    float input = parameters[dialParameterIndices[flatDial]];
    float response = 1.0f;
    uint dialType = dialTypes[flatDial];
    if( dialType == kDialTypeNorm ){
      response = input;
    }
    else if( dialType == kDialTypeCompactSpline ){
      uint splineOffset = dialSplineOffsets[flatDial];
      uint splineSize = dialSplineSizes[flatDial];
      response = evaluate_compact_spline(
          input,
          dialAllowExtrapolation[flatDial] != 0,
          splineData + splineOffset,
          splineSize - 2
      );
    }
    else if( dialType == kDialTypeUniformSpline ){
      uint splineOffset = dialSplineOffsets[flatDial];
      uint splineSize = dialSplineSizes[flatDial];
      response = evaluate_uniform_spline(
          input,
          dialAllowExtrapolation[flatDial] != 0,
          splineData + splineOffset,
          splineSize
      );
    }
    else if( dialType == kDialTypeMonotonicSpline ){
      uint splineOffset = dialSplineOffsets[flatDial];
      uint splineSize = dialSplineSizes[flatDial];
      response = evaluate_monotonic_spline(
          input,
          dialAllowExtrapolation[flatDial] != 0,
          splineData + splineOffset,
          splineSize - 2
      );
    }
    response = max(response, dialMinResponses[flatDial]);
    response = min(response, dialMaxResponses[flatDial]);
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
