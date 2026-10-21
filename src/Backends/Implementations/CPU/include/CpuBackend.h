#ifndef GUNDAM_CPU_BACKEND_H
#define GUNDAM_CPU_BACKEND_H

#include "EngineView.h"
#include "EventStateCache.h"
#include "PropagationResult.h"
#include "BackendTypes.h"
#include "Backend.h"
#include "CpuBackendProfiler.h"
#include "PropagationInputs.h"
#include "GenericToolbox.Time.h"
#include "GenericToolbox.Thread.h"

#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <vector>

namespace Backends {

#define ENUM_NAME CpuWeightScheduling
#define ENUM_FIELDS \
ENUM_FIELD( Fixed, 0 ) \
ENUM_FIELD( Dynamic )
#include "GenericToolbox.MakeEnum.h"

#define ENUM_NAME CpuInputEvaluation
#define ENUM_FIELDS \
ENUM_FIELD( Shared, 0 ) \
ENUM_FIELD( PerDial )
#include "GenericToolbox.MakeEnum.h"

#define ENUM_NAME CpuCacheTraversal
#define ENUM_FIELDS \
ENUM_FIELD( Auto, 0 ) \
ENUM_FIELD( Indexed )
#include "GenericToolbox.MakeEnum.h"

  class CpuBackend : public Backend {
  public:
    CpuBackend() = default;

    [[nodiscard]] std::string getName() const override { return "CPU"; }
    [[nodiscard]] BackendCapabilities getCapabilities() const override;
    [[nodiscard]] PropagationStatus getStatus(const PropagationToken& token_) const override;
    [[nodiscard]] const EngineView& getEngineView() const override { return *_engineView_; }

    void build(const EngineView& engineView_) override;
    void buildShared(std::shared_ptr<const EngineView> engineView_) override;
    PropagationToken requestPropagation(const PropagationInputs& inputs_) override;

    bool isReady(const PropagationToken& token_) const override;
    void wait(const PropagationToken& token_) override;
    void materialize(const PropagationToken& token_, OutputRequest output_) override;
    [[nodiscard]] double getLikelihood(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getEventWeightsHostView(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getHistogramSumsHostView(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getHistogramSumSquaresHostView(const PropagationToken& token_) const override;
    [[nodiscard]] const EventState& getEventStateHostView(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getOutsideSumsHostView(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getOutsideSumSquaresHostView(const PropagationToken& token_) const override;
    [[nodiscard]] BackendTimingSummary getLastTimingSummary() const override { return _lastTiming_; }
    [[nodiscard]] BackendTimingTotals getTimingTotals() const override { return _timingTotals_; }

    // Diagnostic API: call between synchronous propagations. Only the work
    // ranges/kernel selection change; model data and response caches are kept.
    void setDiagnosticWeightScheduling(CpuWeightScheduling scheduling_, std::size_t blocksPerThread_);
    void setDiagnosticWeightWorkers(std::size_t workers_, bool timing_);
    void setDiagnosticCacheTraversal(CpuCacheTraversal traversal_);
    void setDiagnosticInputEvaluation(CpuInputEvaluation evaluation_);
    [[nodiscard]] CpuBackendProfiler::Json getWeightSchedulingDiagnostics() const;

  protected:
    void configureImpl() override;

  private:
    using Result = PropagationResult;

    [[nodiscard]] bool isCurrentToken(const PropagationToken& token_) const;
    void initializeThreads();
    void initializeProfiler();
    void initializeDialResponseCache();
    void initializeCachedInputGroups();
    [[nodiscard]] std::size_t getCachedInputStorageBytes() const;
    void initializeWeightExecution();
    void initializeWeightScheduling();
    template<bool allResponsesCached, bool useReweightCap, bool dynamicScheduling>
    void calculateEventWeightsKernel(int iThread_);
    void resetResult();
    void updateCachedDialResponses(const PropagationInputs& inputs_);
    void updateCachedDialResponsesThread(int iThread_);
    void calculateEventWeights(Result& result_, const PropagationInputs& inputs_);
    void calculateEventWeightsThread(int iThread_);
    void updateEventState(const PropagationInputs& inputs_);
    void runEventStateJob(const std::string& jobName_);
    void updateEventVariablesThread(int iThread_);
    void updateEventBinningThread(int iThread_);
    double evalResponse(std::size_t dial_) const;
    void calculateHistograms(Result& result_);
    void calculateHistogramsThread(int iThread_);
    void calculateHistogramsFromEvents(Result& result_, const PropagationInputs& inputs_);
    void calculateLikelihood(Result& result_);

    struct BinningWorker {
      std::vector<std::size_t> candidates{};
      std::vector<std::uint8_t> changedSamples{};
      std::size_t rebinnedEvents{0};
      std::size_t candidatesTested{0};
    };
    std::vector<std::exception_ptr> _workerErrors_{};
    EventState _eventState_{};
    std::vector<BinningIndex> _binningIndices_{};
    std::vector<BinningWorker> _binningWorkers_{};
    std::vector<std::uint8_t> _validBinning_{};
    EventStateCache _eventStateCache_{};
    std::size_t _cachedContributionCount_{0};
    GenericToolbox::Time::Timer _variableStageTimer_{};
    GenericToolbox::Time::Timer _binningStageTimer_{};
    std::shared_ptr<const EngineView> _engineView_{std::make_shared<EngineView>()};
    Result _lastResult_{};
    GenericToolbox::ParallelWorker _threadPool_{};
    Result* _activeResult_{nullptr};
    const PropagationInputs* _activeInputs_{nullptr};
    std::vector<std::vector<std::uint32_t>> _cachedDialsByParameter_{};
    std::vector<std::uint32_t> _dirtyCachedDialIndices_{};
    std::vector<std::uint8_t> _changedCacheParameters_{};
    CpuCacheTraversal _cacheTraversal_{CpuCacheTraversal::Auto};
    std::size_t _parameterDependentCachedDialCount_{0};
    std::size_t _dirtyCachedDialCount_{0};
    bool _usesLinearCacheTraversal_{false};
    std::vector<double> _cachedDialResponses_{};
    std::vector<double> _cachedDialInputs_{};
    struct CachedInputGroup {
      std::size_t firstInput{0};
      double previousValue{0};
      double value{0};
      bool valid{false};
      bool changed{false};
    };
    std::vector<CachedInputGroup> _cachedInputGroups_{};
    std::vector<std::uint32_t> _cachedDialInputGroups_{};
    std::vector<std::vector<std::uint32_t>> _cachedInputGroupsByParameter_{};
    CpuInputEvaluation _inputEvaluation_{CpuInputEvaluation::Shared};
    std::vector<double> _lastParameterValues_{};
    std::vector<std::uint8_t> _isCachedDial_{};
    std::vector<std::uint8_t> _isCachedDialResponseValid_{};
    bool _isDialResponseCachePrimed_{false};
    std::vector<std::vector<double>> _threadHistogramSums_{};
    std::vector<std::vector<double>> _threadHistogramSumSquares_{};
    GenericToolbox::Time::Timer _cachedDialStageTimer_{};
    GenericToolbox::Time::Timer _eventWeightsStageTimer_{};
    GenericToolbox::Time::Timer _histogramStageTimer_{};
    GenericToolbox::Time::Timer _likelihoodHostTimer_{};
    struct EventRange { std::size_t begin{0}; std::size_t end{0}; std::size_t dialReferences{0}; };
    std::vector<EventRange> _weightRanges_{};
    CpuWeightScheduling _weightScheduling_{CpuWeightScheduling::Dynamic};
    std::size_t _weightBlocksPerThread_{2};
    std::size_t _weightWork_{0};
    std::size_t _diagnosticWeightWorkers_{0};
    bool _diagnosticWeightTiming_{false};
    bool _allWeightResponsesCached_{false};
    bool _usesDynamicWeights_{false};
    std::vector<EventRange> _weightBlocks_{};
    alignas(64) std::atomic<std::size_t> _nextWeightBlock_{0};
    struct alignas(64) WeightWorkerStats {
      std::size_t events{0}, blocks{0}, dialReferences{0};
      double wallSeconds{0};
    };
    std::vector<WeightWorkerStats> _weightWorkerStats_{};
    using WeightJob = void (CpuBackend::*)(int);
    WeightJob _weightJob_{nullptr};
    std::size_t _cachedDialCount_{0};
    bool _requiresEventBinning_{false};
    BackendTimingTotals _timingTotals_{};
    BackendTimingSummary _lastTiming_{};
    std::uint64_t _nextTokenId_{1};
    bool _isBuilt_{false};
    std::string _profileOutputPath_{};
    int _profileSampleInterval_{100};
    int _profileMaxSamples_{200};
    std::unique_ptr<CpuBackendProfiler> _profiler_{};
  };

}

#endif // GUNDAM_CPU_BACKEND_H
