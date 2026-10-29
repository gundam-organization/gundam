#ifndef GUNDAM_NATIVE_DIAL_PROGRAM_MODEL_H
#define GUNDAM_NATIVE_DIAL_PROGRAM_MODEL_H
#include "EngineView.h"
#include <utility>
namespace Backends {
  struct NativeCompositeTemplate {
    std::size_t firstInstruction{0}, instructionCount{0}, inputCount{0}, outputSlot{0};
    bool isChain{false};
    std::size_t scratchSize() const { return inputCount + instructionCount; }
  };
  struct NativeCompositeInstance {
    std::size_t programIndex{0}, firstNodeData{0};
  };
  // CPU/CUDA share immutable templates; per-node data is just a payload/binding offset.
  // The source EngineView owns dialInputs/payloads and must outlive this model.
  struct NativeDialProgramModel {
    const BackendDialInputDescriptor* dialInputs{nullptr};
    const double* payloads{nullptr};
    std::vector<BackendDialDescriptor> dials;
    std::vector<std::pair<std::size_t, std::size_t>> rootFormulaData;
    std::vector<FormulaProgram> formulaPrograms;
    std::vector<FormulaInput> formulaInputs;
    std::vector<NativeCompositeTemplate> compositePrograms;
    std::vector<NativeCompositeInstance> compositeInstances;
    std::vector<CompositeInstruction> compositeInstructions;
    std::vector<BackendDialInputDescriptor> compositeInputs;
    std::vector<std::size_t> compositeNodeData;
    void build(const PropagationView& source_, const std::vector<BackendDialDescriptor>& roots_);
  };
}
#endif
