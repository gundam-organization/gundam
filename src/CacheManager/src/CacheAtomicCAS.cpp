#include "CacheAtomicCAS.h"
#include <mutex>

namespace {
  // Lock the atomic operation.  Only used when builtins are not available.
  std::mutex cacheAtomicCASMutEx;
}

// This is a function to do an atomic compare-and-set.  Compare-and-set is
// fairly common (search the web for a lot of literature) in multi-threaded
// programming where it is used to implement operations like "atomic
// addition".  The AMD64 architecture supports it as a native instruction.  In
// the C/C++ world it is often called compare_exchange, because being different
// is it's own virtue (?).
//
// The compare-and-set operation takes three values.  The variable to be
// changed, the expected value of the variable (prior to the change), and the
// value the variable should hold after the change.  The variable to be
// changed will be updated if it has a value equal to expected.  The return
// value is true if the update succeeded, and false if the updated failed.
// After the call, the expected value will contain the original value of the
// variable.
//
// Example: Implementing atomic add.
//
//  ```
//  double expect = *value;
//  double update;
//  do {update=expect+update;} while (!CacheAtomicCAS(value,&expect,update));
// ```
//
// There is a C++ interface, but it does not work for non-atomic variables
// without some fancy footwork in the call.  This is an easier way for people
// to understand.
int CacheAtomicCAS(double *variable, double *expected, double update) {
#ifdef TEMP_CACHE_ATOMIC_CAS_MANAGED
  #error internal definition TEMP_CACHE_ATOMIC_CAS_MANAGED must be undefined
#undef TEMP_CACHE_ATOMIC_CAS_MANAGED
#endif
#ifndef TEMP_CACHE_ATOMIC_CAS_MANAGED
#ifdef __has_builtin
#if __has_builtin(__atomic_compare_exchange)
  // #warning CacheAtomicCAS -- Using builtin __atomic_compare_exchange
    return __atomic_compare_exchange(
        variable, expected, &update, 0,
        __ATOMIC_SEQ_CST,
        __ATOMIC_SEQ_CST);
#define TEMP_CACHE_ATOMIC_CAS_MANAGED
#endif
#endif
#endif
#ifndef TEMP_CACHE_ATOMIC_CAS_MANAGED
#ifdef __GNUC__
#include <features.h>
#if __GNUC_PREREQ(5,0)
  // #warning CacheAtomicCAS -- Using GCC >5 builtin __atomic_compare_exchange
    return __atomic_compare_exchange(
        variable, expected, &update, 0,
        __ATOMIC_SEQ_CST,
        __ATOMIC_SEQ_CST);
#define TEMP_CACHE_ATOMIC_CAS_MANAGED
#endif
#endif
#endif
#ifndef TEMP_CACHE_ATOMIC_CAS_MANAGED
#ifdef __STDC_VERSION__
  #if __STDC_VERSION__ >= 201112L
// #warning CacheAtomicCAS -- Using C11 atomic_compare_exchange_strong_explicit
    return atomic_compare_exchange_strong_explicit(
        variable, expected, update,
        memory_order_seq_cst,
        memory_order_seq_cst);
#define TEMP_CACHE_ATOMIC_CAS_MANAGED
#endif
#endif
#endif
#ifndef TEMP_CACHE_ATOMIC_CAS_MANAGED
#warning CacheAtomicCAS -- Using MutEx to implement Compare and Set for doubles
  std::lock_guard<std::mutex> lock(cacheAtomicCASMutEx);
  double expectation = *expected;
  *expected = *variable;
  if (*variable == expectation) {
    *variable = update;
    return true;
  }
  return false;
#define TEMP_CACHE_ATOMIC_CAS_MANAGED
#endif
#undef TEMP_CACHE_ATOMIC_CAS_MANAGED
}

// See documentation for CasheAtomicCAS_double
int CacheAtomicCAS(float *variable, float *expected, float update) {
#ifdef TEMP_CACHE_ATOMIC_CAS_MANAGED
  #error internal definition TEMP_CACHE_ATOMIC_CAS_MANAGED must be undefined
#undef TEMP_CACHE_ATOMIC_CAS_MANAGED
#endif
#ifndef TEMP_CACHE_ATOMIC_CAS_MANAGED
#ifdef __has_builtin
#if __has_builtin(__atomic_compare_exchange)
  // #warning CacheAtomicCAS -- Using builtin __atomic_compare_exchange
    return __atomic_compare_exchange(
        variable, expected, &update, 0,
        __ATOMIC_SEQ_CST,
        __ATOMIC_SEQ_CST);
#define TEMP_CACHE_ATOMIC_CAS_MANAGED
#endif
#endif
#endif
#ifndef TEMP_CACHE_ATOMIC_CAS_MANAGED
#ifdef __GNUC__
#include <features.h>
#if __GNUC_PREREQ(5,0)
  // #warning CacheAtomicCAS -- Using GCC >5 builtin __atomic_compare_exchange
    return __atomic_compare_exchange(
        variable, expected, &update, 0,
        __ATOMIC_SEQ_CST,
        __ATOMIC_SEQ_CST);
#define TEMP_CACHE_ATOMIC_CAS_MANAGED
#endif
#endif
#endif
#ifndef TEMP_CACHE_ATOMIC_CAS_MANAGED
#ifdef __STDC_VERSION__
  #if __STDC_VERSION__ >= 201112L
// #warning CacheAtomicCAS -- Using C11 atomic_compare_exchange_strong_explicit
    return atomic_compare_exchange_strong_explicit(
        variable, expected, update,
        memory_order_seq_cst,
        memory_order_seq_cst);
#define TEMP_CACHE_ATOMIC_CAS_MANAGED
#endif
#endif
#endif
#ifndef TEMP_CACHE_ATOMIC_CAS_MANAGED
#warning CacheAtomicCAS -- Using MutEx to implement Compare and Set for floats
  std::lock_guard<std::mutex> lock(cacheAtomicCASMutEx);
  float expectation = *expected;
  *expected = *variable;
  if (*variable == expectation) {
    *variable = update;
    return true;
  }
  return false;
#define TEMP_CACHE_ATOMIC_CAS_MANAGED
#endif
#undef TEMP_CACHE_ATOMIC_CAS_MANAGED
}

// See documentation for CacheAtomicCAS_double
int CacheAtomicCAS(int *variable, int *expected, int update) {
#ifdef TEMP_CACHE_ATOMIC_CAS_MANAGED
  #error internal definition TEMP_CACHE_ATOMIC_CAS_MANAGED must be undefined
#undef TEMP_CACHE_ATOMIC_CAS_MANAGED
#endif
#ifndef TEMP_CACHE_ATOMIC_CAS_MANAGED
#ifdef __has_builtin
#if __has_builtin(__atomic_compare_exchange)
  // #warning CacheAtomicCAS -- Using builtin __atomic_compare_exchange
    return __atomic_compare_exchange(
        variable, expected, &update, 0,
        __ATOMIC_SEQ_CST,
        __ATOMIC_SEQ_CST);
#define TEMP_CACHE_ATOMIC_CAS_MANAGED
#endif
#endif
#endif
#ifndef TEMP_CACHE_ATOMIC_CAS_MANAGED
#ifdef __GNUC__
#include <features.h>
#if __GNUC_PREREQ(5,0)
  // #warning CacheAtomicCAS -- Using GCC >5 builtin __atomic_compare_exchange
    return __atomic_compare_exchange(
        variable, expected, &update, 0,
        __ATOMIC_SEQ_CST,
        __ATOMIC_SEQ_CST);
#define TEMP_CACHE_ATOMIC_CAS_MANAGED
#endif
#endif
#endif
#ifndef TEMP_CACHE_ATOMIC_CAS_MANAGED
#ifdef __STDC_VERSION__
  #if __STDC_VERSION__ >= 201112L
// #warning CacheAtomicCAS -- Using C11 atomic_compare_exchange_strong_explicit
    return atomic_compare_exchange_strong_explicit(
        variable, expected, update,
        memory_order_seq_cst,
        memory_order_seq_cst);
#define TEMP_CACHE_ATOMIC_CAS_MANAGED
#endif
#endif
#endif
#ifndef TEMP_CACHE_ATOMIC_CAS_MANAGED
#warning CacheAtomicCAS -- Using MutEx to implement Compare and Set for integers
  std::lock_guard<std::mutex> lock(cacheAtomicCASMutEx);
  int expectation = *expected;
  *expected = *variable;
  if (*variable == expectation) {
    *variable = update;
    return true;
  }
  return false;
#define TEMP_CACHE_ATOMIC_CAS_MANAGED
#endif
#undef TEMP_CACHE_ATOMIC_CAS_MANAGED
}
