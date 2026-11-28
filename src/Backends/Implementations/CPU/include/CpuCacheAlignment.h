#ifndef GUNDAM_CPU_CACHE_ALIGNMENT_H
#define GUNDAM_CPU_CACHE_ALIGNMENT_H

#include <cstddef>
#include <new>

namespace Backends {
  // Separate independently written data to avoid false sharing. This is a
  // compiler-target recommendation, not necessarily the physical cache-line size.
#if defined(__cpp_lib_hardware_interference_size)
  inline constexpr std::size_t cpuDestructiveInterferenceSize = std::hardware_destructive_interference_size;
#elif defined(__APPLE__) && defined(__aarch64__)
  // Older standard libraries do not expose the recommendation. Apple Silicon
  // uses 128-byte cache lines; retain that separation on those toolchains.
  inline constexpr std::size_t cpuDestructiveInterferenceSize = 128;
#else
  // Conservative fallback for toolchains without this C++17 library feature,
  // covering the common 64-byte cache lines on x86 targets.
  inline constexpr std::size_t cpuDestructiveInterferenceSize = 64;
#endif
}

#endif
