#ifndef GUNDAM_BACKEND_ENGINE_DESCRIPTORS_H
#define GUNDAM_BACKEND_ENGINE_DESCRIPTORS_H

#include <cstddef>
#include <cstdint>

namespace Backends {

  // POD descriptors shared by host propagation and device-capable semantics.
  // They intentionally contain no owning containers or host-only behaviour.
  enum class BackendDialType : std::uint8_t {
    Norm = 0,
    Shift,
    CompactSpline,
    UniformSpline,
    MonotonicSpline,
    GeneralSpline,
    Graph
  };

  struct BackendDialInputDescriptor {
    std::size_t parameterIndex{std::size_t(-1)};
    bool useMirror{false};
    double mirrorMin{0};
    double mirrorRange{0};
  };

  struct BackendDialDescriptor {
    BackendDialType type{BackendDialType::Norm};
    std::size_t firstInput{0};
    std::size_t inputCount{0};
    std::size_t payloadOffset{0};
    std::size_t payloadSize{0};
    bool allowExtrapolation{false};
    double minResponse{0};
    double maxResponse{0};
    bool hasMinResponse{false};
    bool hasMaxResponse{false};
  };

  struct BackendEventWeightDescriptor {
    double baseWeight{1};
    std::size_t firstDial{0};
    std::size_t dialCount{0};
  };

}

#endif // GUNDAM_BACKEND_ENGINE_DESCRIPTORS_H
