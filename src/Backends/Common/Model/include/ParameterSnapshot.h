#ifndef GUNDAM_BACKEND_PARAMETER_SNAPSHOT_H
#define GUNDAM_BACKEND_PARAMETER_SNAPSHOT_H

#include <cstdint>
#include <vector>

namespace Backends {

  struct ParameterSnapshot {
    std::vector<double> values{};
    std::uint64_t version{0};

    [[nodiscard]] bool empty() const { return values.empty(); }
  };

}

#endif // GUNDAM_BACKEND_PARAMETER_SNAPSHOT_H
