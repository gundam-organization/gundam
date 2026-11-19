#ifndef GUNDAM_BACKEND_SEMANTICS_QUALIFIERS_H
#define GUNDAM_BACKEND_SEMANTICS_QUALIFIERS_H

#include <cassert>

#if defined(__CUDACC__)
#define GUNDAM_BACKEND_HOST_DEVICE __host__ __device__
#define GUNDAM_BACKEND_FORCE_INLINE __forceinline__
#else
#define GUNDAM_BACKEND_HOST_DEVICE
#define GUNDAM_BACKEND_FORCE_INLINE inline
#endif

// Policy adapters must disappear even when they wrap a large kernel.
#if defined(__CUDACC__)
#define GUNDAM_BACKEND_ALWAYS_INLINE __forceinline__
#elif defined(__GNUC__) || defined(__clang__)
#define GUNDAM_BACKEND_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define GUNDAM_BACKEND_ALWAYS_INLINE inline
#endif

#ifndef GUNDAM_BACKEND_SEMANTICS_ASSERT
#define GUNDAM_BACKEND_SEMANTICS_ASSERT(condition) assert(condition)
#endif

#endif // GUNDAM_BACKEND_SEMANTICS_QUALIFIERS_H
