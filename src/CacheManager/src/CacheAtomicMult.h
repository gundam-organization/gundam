#ifndef CacheAtomicMult_h_seen
#define CacheAtomicMult_h_seen

#ifndef HEMI_DEV_CODE
#include "CacheAtomicCAS.h"
#endif

namespace {
    /// Do an atomic multiplication for doubles.  On the GPU this uses
    /// compare-and-set.  On the CPU, this leverages the C atomic operations.
    HEMI_DEV_CALLABLE_INLINE
    double CacheAtomicMult(double* address, const double v) {
#ifndef HEMI_DEV_CODE
        // C++ is a little funky on it's support for atomic operations, so
        // give it a little help.
        double expect = *address;
        double update;
        do {
            update = expect*v;
        } while (not CacheAtomicCAS_double(address,&expect,update));
        return expect;
#else
        // When using CUDA use atomic compare-and-set to do an atomic
        // multiplication.  This only sets the result if the value at
        // address_as_ull is equal to "assumed" after the multiplication.  The
        // comparison is done with integers because of CUDA.  The return value
        // of atomicCAS is the original value at address_as_ull, so if it's
        // equal to "assumed", then nothing changed the value at the address
        // while the multiplication was being done.  If something changed the
        // value at the address "old" will not equal "assumed", and then retry
        // the multiplication.
        unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull;
        unsigned long long int assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull,
                            assumed,
                            __double_as_longlong(
                                v *  __longlong_as_double(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN
            // (since NaN != NaN)
        } while (not (assumed == old));
        return __longlong_as_double(old);
#endif
    }
}
#endif
// Local Variables:
// mode:c++
// c-basic-offset:4
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
