#ifndef GUNDAM_CUDA_EVENT_UPDATES_H
#define GUNDAM_CUDA_EVENT_UPDATES_H

#include <array>
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
  struct CudaEventSelection {
    std::vector<std::uint32_t> groups{};
    std::size_t affectedEvents{0};
    bool complete{false};
  };

  // Bound to one immutable model index. Keys are exact changed-parameter masks
  // plus the selection limit; values never depend on the parameter values.
  // Rebinding after model replacement clears every cached decision.
  class CudaEventSelectionCache {
  public:
    void bind(const CudaEventUpdateIndex& index_);
    const CudaEventSelection& select(const std::vector<double>& previous_,
        const std::vector<double>& current_, std::size_t maxGroups_);
    [[nodiscard]] const CudaEventSelection& result() const { return _entries_[_current_].selection; }
    [[nodiscard]] bool wasHit() const { return _wasHit_; }
  private:
    struct Entry {
      std::vector<std::uint64_t> signature{};
      CudaEventSelection selection{};
      std::size_t maxGroups{0};
      bool valid{false};
    };
    const CudaEventUpdateIndex* _index_{nullptr};
    CudaEventUpdatePlan _work_{};
    std::array<Entry, 8> _entries_{};
    std::vector<std::uint64_t> _signature_{};
    std::size_t _next_{0}, _current_{0};
    bool _wasHit_{false};
  };
}
#endif
