#ifndef GUNDAM_DEVICE_PROPAGATION_MODEL_H
#define GUNDAM_DEVICE_PROPAGATION_MODEL_H
#include "DevicePropagationTypes.h"
#include "EngineView.h"
namespace Backends {
  // Device-independent packing of observables and sparse histogram geometry.
  // Metal/CUDA executors own allocations, queues and execution, not this model.
  struct DevicePropagationModel {
    std::vector<DevicePropagationEvent> events;
    std::vector<DevicePropagationDial> dials;
    std::vector<DeviceVariableResponse> variableResponses;
    std::vector<DeviceBinningSample> samples;
    std::vector<DeviceBinningInterval> intervals;
    std::vector<DeviceBinningBin> bins;
    std::vector<DeviceBinningNode> nodes;
    std::vector<DeviceBinningBounds> bounds;
    std::vector<DeviceIndex> nodeBins, weightDialIndices, variableDialIndices;
    std::vector<ObservableScalar> variables, payloads;
    DeviceIndex contributionCapacity{0};
    DeviceIndex histogramBins{0};
    DeviceIndex histogramChunkCapacity{0};
    void build(const PropagationView& model_);
  };
}
#endif
