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
    void setEnableDialCache(bool enabled_){ if( _enableDialCache_ != enabled_ ){ _enableDialCache_ = enabled_; invalidate(); } }
    void setEnableBinFactorization(bool enabled_){ if( _enableBinFactorization_ != enabled_ ){ _enableBinFactorization_ = enabled_; invalidate(); } }
    [[nodiscard]] std::string getDiagnostics() const;
    [[nodiscard]] std::string getExecutionState() const;
    [[nodiscard]] const std::map<std::string, double>& getDiagnosticStageGpuSeconds() const { return _diagnosticStageGpuSeconds_; }
    void build(const PropagationView& model_, const LikelihoodView& likelihood_ = {});
    void updateDataHistograms(const LikelihoodView& likelihood_);
    void prepareGradient(const EngineView& view_);
    [[nodiscard]] bool hasGradientBuffers() const;
    std::vector<double> getLikelihoodGradient(const EngineView& view_);
    void materializeHistograms(PropagationResult& result_, BackendTimingSummary& timing_);
    void clear();
    void invalidate(){ _cache_.invalidate(); _valid_ = false; }
    void propagate(const PropagationInputs& inputs_, PropagationResult& result_, BackendTimingSummary& timing_);
    void materializeWeights(PropagationResult& result_, BackendTimingSummary& timing_);
    const EventState& materializeState(BackendTimingSummary& timing_);
    [[nodiscard]] BackendDeviceView getDeviceView();

  private:
    enum class Buffer {
      Events, Samples, Dials, DialInputs, DialInputRanges, Targets, Intervals, Bins, Nodes, Bounds, NodeBins, ReusableBins,
      WeightDialIndices, VariableDialIndices, Payloads, Parameters, ExternalWeights, Responses, DialIndices, DialRanges,
      Variables, HostValues, HostEdits, DirtyEvents, HardBins, EventCounts, Contributions, ChangedSamples, Control,
      EventWeights, BinCounts, BinPositions, BinContributions, ChunkOffsets, Chunks, BinFactors, BinFactorOffsets, FactoredDialMask, ResidualSums, ResidualSquares,
      PartialSums, PartialSquares, HistogramSums, HistogramSquares, ReadbackSums, ReadbackSquares,
      ReadbackVariables, ReadbackBins, ReadbackCounts, ReadbackContributions, ReadbackWeights, FormulaPrograms, FormulaInstructions, FormulaConstants,
      ProgramInputs, CompositePrograms, CompositeInstructions, ProgramScratch, LikelihoodBins, LikelihoodCount, LikelihoodPartials, LikelihoodTotal, LikelihoodReadback, GradientDials, GradientEvents, GradientBins, GradientIndices, GradientOffsets, GradientOccurrences, GradientResponses, GradientScales, GradientContributions, GradientTotal, GradientReadback, GradientBindings, GradientRanges, GradientInputOffsets, GradientScratch, GradientExternal, GradientWork, Count
    };
    enum class Kernel { Dials, Events, Binning, ClearCounts, CountContributions, IndexChunks, Scatter, Reduce, Finish, ReduceParallel, LikelihoodPartials, LikelihoodFinish, GradientDials, GradientBins, GradientEvents, GradientReduce, ApplyBinFactors, Count };
    id<MTLBuffer> buffer(Buffer buffer_) const { return _buffers_[std::size_t(buffer_)]; }
    struct DialGroup { std::size_t dependency, first, count; bool external; };
    struct MultiDialGroup { std::vector<std::size_t> parameters; std::size_t first, count; };
    std::vector<MultiDialGroup> _multiDialGroups_{};
    void buildDialCache();
    bool usesBinFactorization() const;
    void propagateImpl(const PropagationInputs& inputs_, PropagationResult& result_, BackendTimingSummary& timing_, bool factor_);
    std::size_t prepareDialCache(const PropagationInputs& inputs_, bool& updateWeights_, bool& updateResidualWeights_);
    void encodeEventWeights(id<MTLComputeCommandEncoder> encoder_);
    void ensureEventWeights();
    void clearBuffers();
    void dispatch(id<MTLComputeCommandEncoder> encoder_, Kernel kernel_, std::size_t count_, std::initializer_list<Buffer> buffers_);
    void copy(id<MTLBlitCommandEncoder> encoder_, Buffer source_, Buffer destination_, std::size_t bytes_);
    id<MTLDevice> _device_;
    id<MTLCommandQueue> _queue_;
    std::array<id<MTLBuffer>, std::size_t(Buffer::Count)> _buffers_{};
    std::array<id<MTLComputePipelineState>, std::size_t(Kernel::Count)> _pipelines_{};
    const PropagationView* _model_{nullptr};
    std::size_t _dialCount_{0};
    DevicePropagationModel _packed_{};
    EventStateCache _cache_{};
    EventState _state_{};
    std::vector<std::uint64_t> _externalGenerations_{};
    std::uint64_t _materializedRevision_{0};
    bool _valid_{false};
    bool _deviceLikelihood_{false};
    LikelihoodWorkLayout _likelihoodWork_{};
    GradientWork<ObservableScalar> _gradientWork_{};
    std::vector<std::size_t> _producerDials_{};
    std::vector<std::size_t> _producerInputCounts_{};
    std::vector<std::vector<double>> _producerGradients_{};
    bool _producerGradientsValid_{false};
    bool _lastUpdatedPrediction_{true};
    bool _enableDiagnosticStageProfiling_{false};
    bool _enableDialCache_{true};
    bool _enableBinFactorization_{true};
    bool _eventWeightsValid_{false};
    bool _residualHistogramsValid_{false};
    bool _lastReusedResidualHistograms_{false};
    std::size_t _factoredBinCount_{0}, _multiFactorBinCount_{0}, _factoredParameterCount_{0}, _binFactorReuseCount_{0};
    std::size_t _factoredExternalBlockCount_{0};
    std::vector<DialGroup> _dialGroups_{};
    std::vector<double> _cachedParameters_{};
    std::vector<std::uint64_t> _cachedExternalGenerations_{};
    std::vector<std::size_t> _weightParameters_{}, _weightExternalBlocks_{};
    std::vector<bool> _residualWeightParameters_{}, _residualWeightExternalBlocks_{};
    std::size_t _lastEvaluatedDials_{0};
    bool _lastUpdatedWeights_{true}, _lastRebuiltIndex_{true};
    std::map<std::string, double> _diagnosticStageGpuSeconds_{};
  };
}
#endif
