#ifndef GUNDAM_MPS_EVENT_PROPAGATION_H
#define GUNDAM_MPS_EVENT_PROPAGATION_H

#include "DevicePropagationModel.h"
#include "GradientTypes.h"
#include "EventStateCache.h"
#include "PropagationResult.h"

#import <Metal/Metal.h>

#include <array>
#include <map>
#include <string>

namespace Backends {
  // Metal executor for resident event observables and sparse contributions.
  // Invalidation, geometry construction and scalar semantics are shared.
  class MpsEventPropagation {
  public:
    MpsEventPropagation(id<MTLDevice> device_, id<MTLCommandQueue> queue_, id<MTLLibrary> library_);
    ~MpsEventPropagation();
    MpsEventPropagation(const MpsEventPropagation&) = delete;
    MpsEventPropagation& operator=(const MpsEventPropagation&) = delete;
    void setDiagnosticStageProfiling(bool enabled_){ _enableDiagnosticStageProfiling_ = enabled_; }
    void setEnableDialCache(bool enabled_){ _enableDialCache_ = enabled_; }
    [[nodiscard]] std::string getDiagnostics() const;
    [[nodiscard]] std::string getExecutionState() const;
    [[nodiscard]] const std::map<std::string, double>& getDiagnosticStageGpuSeconds() const { return _diagnosticStageGpuSeconds_; }
    void build(const PropagationView& model_, const LikelihoodView& likelihood_ = {});
    std::vector<double> getLikelihoodGradient(const EngineView& view_);
    void materializeHistograms(PropagationResult& result_, BackendTimingSummary& timing_);
    void clear();
    void invalidate(){ _cache_.invalidate(); _valid_ = false; }
    void propagate(const PropagationInputs& inputs_, PropagationResult& result_, BackendTimingSummary& timing_);
    void materializeWeights(PropagationResult& result_, BackendTimingSummary& timing_);
    const EventState& materializeState(BackendTimingSummary& timing_);
    [[nodiscard]] BackendDeviceView getDeviceView() const;

  private:
    enum class Buffer {
      Events, Samples, Dials, DialInputs, DialInputRanges, Targets, Intervals, Bins, Nodes, Bounds, NodeBins, ReusableBins,
      WeightDialIndices, VariableDialIndices, Payloads, Parameters, ExternalWeights, Responses, DialIndices, DialRanges,
      Variables, HostValues, HostEdits, DirtyEvents, HardBins, EventCounts, Contributions, ChangedSamples, Control,
      EventWeights, BinCounts, BinPositions, BinContributions, ChunkOffsets, Chunks,
      PartialSums, PartialSquares, HistogramSums, HistogramSquares, ReadbackSums, ReadbackSquares,
      ReadbackVariables, ReadbackBins, ReadbackCounts, ReadbackContributions, ReadbackWeights, FormulaPrograms, FormulaInstructions, FormulaConstants,
      ProgramInputs, CompositePrograms, CompositeInstructions, ProgramScratch, LikelihoodBins, LikelihoodCount, LikelihoodPartials, LikelihoodTotal, LikelihoodReadback, GradientDials, GradientEvents, GradientBins, GradientPayloads, GradientIndices, GradientOffsets, GradientOccurrences, GradientResponses, GradientScales, GradientContributions, GradientTotal, GradientReadback, GradientWork, Count
    };
    enum class Kernel { Dials, Events, Binning, ClearCounts, CountContributions, IndexChunks, Scatter, Reduce, Finish, ReduceParallel, LikelihoodPartials, LikelihoodFinish, GradientDials, GradientBins, GradientEvents, GradientReduce, Count };
    id<MTLBuffer> buffer(Buffer buffer_) const { return _buffers_[std::size_t(buffer_)]; }
    struct DialGroup { std::size_t dependency, first, count; bool external; };
    struct MultiDialGroup { std::vector<std::size_t> parameters; std::size_t first, count; };
    std::vector<MultiDialGroup> _multiDialGroups_{};
    void buildDialCache();
    std::size_t prepareDialCache(const PropagationInputs& inputs_, bool& updateWeights_);
    void clearBuffers();
    void dispatch(id<MTLComputeCommandEncoder> encoder_, Kernel kernel_, std::size_t count_, std::initializer_list<Buffer> buffers_);
    void copy(id<MTLBlitCommandEncoder> encoder_, Buffer source_, Buffer destination_, std::size_t bytes_);
    id<MTLDevice> _device_;
    id<MTLCommandQueue> _queue_;
    std::array<id<MTLBuffer>, std::size_t(Buffer::Count)> _buffers_{};
    std::array<id<MTLComputePipelineState>, std::size_t(Kernel::Count)> _pipelines_{};
    const PropagationView* _model_{nullptr};
    DevicePropagationModel _packed_{};
    EventStateCache _cache_{};
    EventState _state_{};
    std::vector<std::uint64_t> _externalGenerations_{};
    std::uint64_t _materializedRevision_{0};
    bool _valid_{false};
    bool _deviceLikelihood_{false};
    LikelihoodWorkLayout _likelihoodWork_{};
    GradientWork<ObservableScalar> _gradientWork_{};
    bool _lastUpdatedPrediction_{true};
    bool _enableDiagnosticStageProfiling_{false};
    bool _enableDialCache_{true};
    std::vector<DialGroup> _dialGroups_{};
    std::vector<double> _cachedParameters_{};
    std::vector<std::uint64_t> _cachedExternalGenerations_{};
    std::vector<std::size_t> _weightParameters_{}, _weightExternalBlocks_{};
    std::size_t _lastEvaluatedDials_{0};
    bool _lastUpdatedWeights_{true}, _lastRebuiltIndex_{true};
    std::map<std::string, double> _diagnosticStageGpuSeconds_{};
  };
}
#endif
