#ifndef GUNDAM_DEVICE_PROPAGATION_MODEL_H
#define GUNDAM_DEVICE_PROPAGATION_MODEL_H
#include "DevicePropagationTypes.h"
#include "EngineView.h"

#include <functional>
namespace Backends {
  // Device-independent packing of observables and sparse histogram geometry.
  // Metal/CUDA executors own allocations, queues and execution, not this model.
  struct DevicePropagationModel {
    std::vector<DevicePropagationEvent> events;
    std::vector<DevicePropagationDial> dials;
    std::vector<DeviceDialInput> dialInputs;
    std::vector<DeviceDialInputRange> dialInputRanges;
    std::vector<DeviceVariableResponse> variableResponses;
    std::vector<DeviceBinningSample> samples;
    std::vector<DeviceBinningInterval> intervals;
    std::vector<DeviceBinningBin> bins;
    std::vector<DeviceBinningNode> nodes;
    std::vector<DeviceBinningBounds> bounds;
    std::vector<DeviceIndex> nodeBins, reusableBins, weightDialIndices, variableDialIndices;
    std::vector<ObservableScalar> variables, payloads;
    std::vector<DeviceFormulaProgram> formulaPrograms;
    std::vector<FormulaInstruction> formulaInstructions;
    std::vector<ObservableScalar> formulaConstants;
    std::vector<DeviceProgramInput> programInputs;
    std::vector<DeviceCompositeProgram> compositePrograms;
    std::vector<DeviceCompositeInstruction> compositeInstructions;
    std::size_t programScratchSize{0};
    DeviceIndex contributionCapacity{0};
    DeviceIndex histogramBins{0};
    DeviceIndex histogramChunkCapacity{0};
    enum class Section { Dials, Payloads, Variables, DialIndices, Targets, Binning, Events };
    // Each callback may consume its completed section. Subsequent sections
    // use source data and offsets, never the contents of emitted vectors.
    void build(const PropagationView& model_, const std::function<void(Section)>& emit_ = {});
  };
}
#endif
