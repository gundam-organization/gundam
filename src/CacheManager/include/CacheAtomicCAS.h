#ifndef CacheAtomicCAS_h_SEEN
#define CacheAtomicCAS_h_SEEN

// Implement a "portable" atomic compare-and-set for various native types.
// This hides the details, and might depend on compiler built-in methods..
// The semantics are similar to the C11 atomic_compare_exchange_weak_explicit
// function.  If the value pointed to by "variable" is equal to the value
// pointed to by "expected", then set the value pointed to by "variable" to
// "update".
int CacheAtomicCAS(double *variable, double *expected, double update);
int CacheAtomicCAS(float *variable, float *expected, float update);
int CacheAtomicCAS(int *variable, int *expected, int update);

#endif
