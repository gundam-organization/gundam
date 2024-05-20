#include <stdatomic.h>
#include "CacheAtomicCAS.h"

// This is a C11 function to do the atomic CAS.  There is a C++ interface, but
// it does not work for non-atomic variables without some fancy footwork in
// the call.  This is an easier way for people to understand.
int CacheAtomicCAS_double(double *variable, double *expected, double update) {
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#warning CacheAtomicCAS -- Using C11 atomic_compare_exchange_weak_explicit
     return atomic_compare_exchange_weak_explicit(
          variable, expected, update,
          memory_order_seq_cst,
          memory_order_seq_cst);
#elif defined(__has_builtin) && __has_builtin(__atomic_compare_exchange)
#warning CacheAtomicCAS -- Using builtin __atomic_compare_exchange
     return __atomic_compare_exchange(
          variable, expected, &update, 0,
          __ATOMIC_ACQUIRE,
          __ATOMIC_ACQUIRE);
#elif __GNUC_PREREQ(5,0)
#warning CacheAtomicCAS -- Using GCC >5 builtin __atomic_compare_exchange
     return __atomic_compare_exchange(
          variable, expected, &update, 0,
          __ATOMIC_ACQUIRE,
          __ATOMIC_ACQUIRE);
#else
#ifdef CACHE_MANAGER_ATOMIC_CAS_NOT_ATOMIC
#error Atomic compare and exchange are not supported by this compiler version
#endif
     // Always "succeed" when atomic operations are not supported.  Notice
     // that this is flagged as a compilation error, so
     // CACHE_MANAGER_ATOMIC_CAS_NOT_ATOMIC will need to be defined during
     // compilation.
     return true;
#endif
}

// See documentation for CasheAtomicCAS_double
int CacheAtomicCAS_float(float *variable, float *expected, float update) {
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
     return atomic_compare_exchange_weak_explicit(
          variable, expected, update,
          memory_order_seq_cst,
          memory_order_seq_cst);
#elif defined(__has_builtin) && __has_builtin(__atomic_compare_exchange)
     return __atomic_compare_exchange(
          variable, expected, &update, 0,
          __ATOMIC_ACQUIRE,
          __ATOMIC_ACQUIRE);
#elif __GNUC_PREREQ(5,0)
     return __atomic_compare_exchange(
          variable, expected, &update, 0,
          __ATOMIC_ACQUIRE,
          __ATOMIC_ACQUIRE);
#else
#ifdef CACHE_MANAGER_ATOMIC_CAS_NOT_ATOMIC
#error Atomic compare and exchange are not supported by this compiler version
#endif
     // Always "succeed" when atomic operations are not supported.  Notice
     // that this is flagged as a compilation error, so
     // CACHE_MANAGER_ATOMIC_CAS_NOT_ATOMIC will need to be defined during
     // compilation.
     return true;
#endif
}

// See documentation for CacheAtomicCAS_double
int CacheAtomicCAS_int(int *variable, int *expected, int update) {
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
     return atomic_compare_exchange_weak_explicit(
          variable, expected, update,
          memory_order_seq_cst,
          memory_order_seq_cst);
#elif defined(__has_builtin) && __has_builtin(__atomic_compare_exchange)
     return __atomic_compare_exchange(
          variable, expected, &update, 0,
          __ATOMIC_ACQUIRE,
          __ATOMIC_ACQUIRE);
#elif __GNUC_PREREQ(5,0)
     return __atomic_compare_exchange(
          variable, expected, &update, 0,
          __ATOMIC_ACQUIRE,
          __ATOMIC_ACQUIRE);
#else
#ifdef CACHE_MANAGER_ATOMIC_CAS_NOT_ATOMIC
#error Atomic compare and exchange are not supported by this compiler version
#endif
     // Always "succeed" when atomic operations are not supported.  Notice
     // that this is flagged as a compilation error, so
     // CACHE_MANAGER_ATOMIC_CAS_NOT_ATOMIC will need to be defined during
     // compilation.
     return true;
#endif
}
