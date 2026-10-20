#ifndef GUNDAM_CPU_BACKEND_PROFILER_H
#define GUNDAM_CPU_BACKEND_PROFILER_H

#include "BackendTypes.h"
#include "GenericToolbox.Json.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace Backends {
  // Optional, sampled diagnostics. Worker records are disjoint and only read
  // by the caller after the thread pool has joined the corresponding job.
  class CpuBackendProfiler {
  public:
    enum class Stage { Cache, Weights, Histograms };
    using Clock = std::chrono::steady_clock;
    using Json = GenericToolbox::Json::JsonType;
    struct Snapshot {
      Clock::time_point wall{};
      double cpuSeconds{-1};
      long minorFaults{-1}, majorFaults{-1}, voluntarySwitches{-1}, involuntarySwitches{-1};
      int cpu{-1};
    };
    struct alignas(64) WorkerRecord {
      Snapshot begin{}, end{};
      std::size_t evaluatedDials{0}, unchangedInputs{0};
      std::array<std::size_t, 8> evaluatedDialTypes{};
      bool ran{false};
    };
    class WorkerScope {
    public:
      WorkerScope(CpuBackendProfiler* profiler_, Stage stage_, int thread_);
      ~WorkerScope();
      WorkerScope(const WorkerScope&) = delete;
      WorkerScope& operator=(const WorkerScope&) = delete;
      WorkerRecord* get() { return _record_; }
    private:
      WorkerRecord* _record_{nullptr};
    };

    CpuBackendProfiler(const std::string& path_, std::size_t interval_, std::size_t maxSamples_,
                       std::size_t threads_, Json metadata_);
    void begin(std::uint64_t call_, const std::vector<double>& parameters_,
               const std::vector<double>& previous_, bool primed_);
    void beginStage(Stage stage_);
    void endStage(Stage stage_);
    void cachePrepared(std::size_t dirtyDials_);
    void end(const BackendTimingSummary& timing_);
    [[nodiscard]] bool isActive() const { return _active_; }

  private:
    static Snapshot snapshot(bool thread_);
    void write(Json record_);
    std::ofstream _output_{};
    std::string _runId_{};
    std::vector<std::string> _cgroupCpuStatPaths_{};
    std::size_t _interval_{100}, _maxSamples_{200}, _samples_{0};
    bool _active_{false};
    std::uint64_t _call_{0};
    Snapshot _begin_{};
    Clock::time_point _created_{Clock::now()};
    double _previousExportSeconds_{0};
    std::vector<std::size_t> _changedParameters_{};
    bool _cachePrimed_{false};
    std::size_t _dirtyDials_{0};
    double _cachePreparationSeconds_{0};
    std::array<std::vector<WorkerRecord>, 3> _workers_{};
    std::array<Clock::time_point, 3> _stageBegin_{};
    std::array<double, 3> _stageSeconds_{};
  };
}
#endif
