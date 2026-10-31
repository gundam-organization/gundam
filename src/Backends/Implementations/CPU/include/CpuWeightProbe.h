#ifndef GUNDAM_CPU_WEIGHT_PROBE_H
#define GUNDAM_CPU_WEIGHT_PROBE_H

#include "GenericToolbox.Json.h"

#include <array>
#include <chrono>
#include <cstdint>

namespace Backends {
  // Optional per-worker, per-weight-job counters. No model data are copied.
  // Construct/read on the owner thread, begin/end on the worker between joins.
  class CpuWeightProbe {
  public:
    using Json = GenericToolbox::Json::JsonType;
    CpuWeightProbe() = default;
    ~CpuWeightProbe();
    CpuWeightProbe(const CpuWeightProbe&) = delete;
    CpuWeightProbe& operator=(const CpuWeightProbe&) = delete;
    void begin();
    void end();
    [[nodiscard]] Json report() const;

  private:
    struct Reading { std::uint64_t value{0}, enabled{0}, running{0}; };
    struct Counter {
      int fd{-1}, error{0};
      bool active{false}, valid{false};
      Reading before{}, after{};
    };
    struct Snapshot {
      std::chrono::steady_clock::time_point wall{};
      double cpuSeconds{-1};
      long voluntarySwitches{-1}, involuntarySwitches{-1}, minorFaults{-1}, majorFaults{-1};
      int cpu{-1};
    };
    static Snapshot snapshot();
    void closeCounters();
    std::array<Counter, 6> _counters_{};
    long _threadId_{-1};
    Snapshot _begin_{}, _end_{};
    bool _measured_{false};
  };
}
#endif
