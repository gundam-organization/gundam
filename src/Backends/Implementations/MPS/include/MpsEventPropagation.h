#ifndef GUNDAM_MPS_EVENT_PROPAGATION_H
#define GUNDAM_MPS_EVENT_PROPAGATION_H

#include "DevicePropagationModel.h"
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
    void setDiagnosticPreviousBinReuse(bool enabled_){ _enablePreviousBinReuse_ = enabled_; }
    void setDiagnosticBinningIntervalPruning(bool enabled_){ _enableBinningIntervalPruning_ = enabled_; }
    void setDiagnosticOptions(bool cacheDials_, bool reuseIndex_, bool parallelHistograms_, bool aggregateCounters_){
      _enableDialCache_ = cacheDials_; _enableIndexReuse_ = reuseIndex_;
      _enableParallelHistograms_ = parallelHistograms_; _enableAggregateCounters_ = aggregateCounters_;
    }
    [[nodiscard]] std::string getDiagnostics() const;
    [[nodiscard]] std::string getExecutionState() const;
    [[nodiscard]] const std::map<std::string, double>& getDiagnosticStageGpuSeconds() const { return _diagnosticStageGpuSeconds_; }
    void build(const PropagationView& model_);
    void clear();
    void invalidate(){ _cache_.invalidate(); _valid_ = false; }
    void propagate(const PropagationInputs& inputs_, PropagationResult& result_, BackendTimingSummary& timing_);
    void materializeWeights(PropagationResult& result_, BackendTimingSummary& timing_);
    const EventState& materializeState(BackendTimingSummary& timing_);
    [[nodiscard]] BackendDeviceView getDeviceView() const;

  private:
    enum class Buffer {
      Events, Samples, Dials, Targets, Intervals, Bins, Nodes, Bounds, NodeBins, ReusableBins,
      WeightDialIndices, VariableDialIndices, Payloads, Parameters, ExternalWeights, Responses, DialIndices, DialRanges,
      Variables, HostValues, HostEdits, DirtyEvents, HardBins, EventCounts, Contributions, ChangedSamples, Control,
      EventWeights, BinCounts, BinOffsets, BinPositions, BinContributions, ChunkOffsets, Chunks,
      PartialSums, PartialSquares, HistogramSums, HistogramSquares, ReadbackSums, ReadbackSquares,
      ReadbackVariables, ReadbackBins, ReadbackCounts, ReadbackContributions, ReadbackWeights, Count
    };
    enum class Kernel { Dials, Events, Binning, ClearCounts, CountContributions, IndexChunks, Scatter, Reduce, Finish, ReduceParallel, Count };
    id<MTLBuffer> buffer(Buffer buffer_) const { return _buffers_[std::size_t(buffer_)]; }
    struct DialGroup { std::size_t dependency, first, count; bool external; };
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
    bool _enableDiagnosticStageProfiling_{false};
    bool _enablePreviousBinReuse_{true};
    bool _enableBinningIntervalPruning_{true};
    bool _enableDialCache_{true}, _enableIndexReuse_{true};
    bool _enableParallelHistograms_{true}, _enableAggregateCounters_{true};
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
