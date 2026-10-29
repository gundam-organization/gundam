#ifndef GUNDAM_LIKELIHOOD_TYPES_H
#define GUNDAM_LIKELIHOOD_TYPES_H
#ifndef __METAL_VERSION__
#include <cstdint>
#define GUNDAM_LLH_INLINE __host__ __device__ inline
#ifndef __CUDACC__
#undef GUNDAM_LLH_INLINE
#define GUNDAM_LLH_INLINE inline
#endif
#else
#define GUNDAM_LLH_INLINE inline
#endif

namespace Backends {
  // Fixed-width, trivially copyable descriptors; independent of JointProbability.
#ifdef __METAL_VERSION__
  using LikelihoodIndex = uint;
#else
  using LikelihoodIndex = std::uint32_t;
#endif
  enum class LikelihoodKind : LikelihoodIndex { Host, Poisson, BarlowBeeston2022 };

  template<typename Scalar> struct LikelihoodOptions {
    LikelihoodKind kind{LikelihoodKind::Host};
    LikelihoodIndex usePoissonLikelihood{0};
    LikelihoodIndex useNominalVariance{0};
    LikelihoodIndex throwIfInfLlh{1};
    Scalar fractionalErrorLimit{};
    Scalar expectedValueMinimum{};
  };
  template<typename Scalar> struct LikelihoodBin {
    LikelihoodIndex globalBin{0};
    LikelihoodIndex ignored{0};
    Scalar data{};
    Scalar nominalVariance{};
    LikelihoodOptions<Scalar> options{};
  };
}
#endif
