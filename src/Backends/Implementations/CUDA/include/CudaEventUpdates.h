#ifndef GUNDAM_CUDA_EVENT_UPDATES_H
#define GUNDAM_CUDA_EVENT_UPDATES_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Backends {
  struct PropagationView;
  struct CudaEventGroupDependency { std::uint32_t group{0}, mask{0}; };

  // Host CSR: physical parameter -> groups of 32 result slots and exact masks.
  // Built only for fixed-state models containing standard scalar dials.
  struct CudaEventUpdateIndex {
    std::vector<std::size_t> offsets{};
    std::vector<CudaEventGroupDependency> dependencies{};
    std::vector<std::size_t> parameterEventCounts{};
    std::size_t eventCount{0}, groupCount{0};
    double buildSeconds{0};
    std::string fallback{};
    void build(const PropagationView& model_, bool hasEventState_, std::size_t maxBytes_ = 256 * 1024 * 1024);
    [[nodiscard]] bool available() const { return not offsets.empty(); }
    [[nodiscard]] std::size_t bytes() const {
      return offsets.size() * sizeof(std::size_t) + dependencies.size() * sizeof(CudaEventGroupDependency)
          + parameterEventCounts.size() * sizeof(std::size_t);
    }
  };

  struct CudaEventUpdatePlan {
    std::vector<std::uint32_t> masks{}, groups{};
    std::size_t affectedEvents{0};
    bool complete{false};
    // Stops as soon as the unique group count exceeds maxGroups_. A partial
    // selection is never used for sparse execution or reported as exact.
    void select(const CudaEventUpdateIndex& index_, const std::vector<double>& previous_,
                const std::vector<double>& current_, std::size_t maxGroups_);
  };
}
#endif
