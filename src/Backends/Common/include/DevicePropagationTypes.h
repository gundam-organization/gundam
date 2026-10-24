#ifndef GUNDAM_DEVICE_PROPAGATION_TYPES_H
#define GUNDAM_DEVICE_PROPAGATION_TYPES_H
#ifndef __METAL_VERSION__
#include "Semantics/BackendResponseSmoothing.h"
#include "DeviceMath/DoubleFloat.h"
#include <cstdint>
#endif
namespace Backends {
#ifdef __METAL_VERSION__
  using DeviceIndex = uint;
#else
  using DeviceIndex = std::uint32_t;
#endif
  using ObservableScalar = GundamDeviceMath::DoubleFloat;
  enum class BackendDialType : DeviceIndex {
    Norm, Shift, CompactSpline, UniformSpline, MonotonicSpline, GeneralSpline, Graph, ExternalWeight, Bilinear, Bicubic, Tabulated, Kriged
  };
  // Wire descriptors: shared definitions, fixed-width fields, no host ownership.
  struct DeviceEventDialRanges {
    DeviceIndex externalOffset{0};
    DeviceIndex externalCount{0};
    DeviceIndex normOffset{0};
    DeviceIndex normCount{0};
    DeviceIndex compactOffset{0};
    DeviceIndex compactCount{0};
    DeviceIndex uniformOffset{0};
    DeviceIndex uniformCount{0};
    DeviceIndex monotonicOffset{0};
    DeviceIndex monotonicCount{0};
    DeviceIndex generalOffset{0};
    DeviceIndex generalCount{0};
    DeviceIndex graphOffset{0};
    DeviceIndex graphCount{0};
  };

  struct DeviceExternalDialOccurrence {
    DeviceIndex weightIndex{0};
    float minResponse{0};
    float maxResponse{0};
    ResponseSmoothing<float> smoothing{};
  };

  struct DeviceNormDialOccurrence {
    DeviceIndex parameterIndex{0};
    float minResponse{1.0F};
    float maxResponse{1.0F};
    ResponseSmoothing<float> smoothing{};
  };

  struct DeviceSplineDialDescriptor {
    DeviceIndex parameterIndex{0};
    DeviceIndex splineOffset{0};
    DeviceIndex splineSize{0};
    DeviceIndex flags{0};
    float minResponse{1.0F};
    float maxResponse{1.0F};
    ResponseSmoothing<float> smoothing{};
  };


  struct DevicePropagationDial {
    DeviceIndex type, parameterIndex, payloadOffset, payloadSize, externalIndex;
    DeviceIndex allowExtrapolation, useMirror;
    ObservableScalar mirrorMin, mirrorRange, minResponse, maxResponse;
    ResponseSmoothing<ObservableScalar> smoothing{};
  };
  struct DeviceDialInput {
    DeviceIndex source, index, useMirror;
    ObservableScalar mirrorMin, mirrorRange;
  };
  struct DeviceDialInputRange { DeviceIndex first, count; };
  struct DeviceVariableResponse {
    DeviceIndex variableIndex;
    ObservableScalar referenceValue;
    DeviceIndex overrideDial, firstMultiplyDial, multiplyDialCount, firstAddDial, addDialCount;
  };
  struct DevicePropagationEvent {
    DeviceIndex firstDial, dialCount, firstVariable, variableCount, firstResponse, responseCount;
    DeviceIndex sample, firstContribution, contributionCapacity;
    int fixedBin;
    float baseWeight;
  };
  struct DeviceBinningSample {
    DeviceIndex binOffset, binCount, firstNode, nodeCount, variableCount;
    DeviceIndex isSmoothed, isDynamic, readHostVariables, outsideBin;
  };
  struct DeviceBinningInterval {
    DeviceIndex variableIndex;
    ObservableScalar lowerBound, upperBound, lowerInverseWidth, upperInverseWidth;
  };
  struct DeviceBinningBin {
    int binIndex;
    DeviceIndex firstInterval, intervalCount, position;
  };
  struct DeviceBinningNode { DeviceIndex firstBounds, firstBin, binCount, skip; };
  struct DeviceBinningBounds { float lower, upper; };
  struct DeviceBinContribution { DeviceIndex event; int binIndex; float fraction; };
  struct DeviceHistogramChunk { DeviceIndex begin, end; };
  struct DevicePropagationDialRange { DeviceIndex firstThread, firstDial, count; };
  struct DevicePropagationControl {
    DeviceIndex error, rebuildIndex, rebinnedEvents, candidates, chunkCount, contributionCount;
    DeviceIndex updateVariables, forceBinning, firstPropagation, editCount;
    DeviceIndex updateWeights, dialRangeCount;
    DeviceIndex eventCount, histogramBins, dialCount, sampleCount;
    DeviceIndex hasReweightCap;
    float maxReweight;
  };
}
#endif
