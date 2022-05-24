#ifndef CacheAtomicAdd_h_seen
#define CacheAtomicAdd_h_seen
namespace {
    /// Do an atomic addition for doubles on the GPU.  On the GPU this
    /// uses compare-and-set.  On the CPU, this is just an addition.  Later
    /// versions of CUDA do support atomicAdd for doubles, but this will work
    /// on earlier versions too.  It's a bit slower, so if we need, there can
    /// some conditional compilation to use the "official" version when it is
    /// available.
    HEMI_DEV_CALLABLE_INLINE
    double CacheAtomicAdd(double* address, const double v) {
#ifndef HEMI_DEV_CODE
        // When this isn't CUDA use a simple addition.
        double old = *address;
        *address = *address + v;
        return old;
#else
        // When using CUDA use atomic compare-and-set to do an atomic
        // addition.  This only sets the result if the value at address_as_ull
        // is equal to "assumed" after the addition.  The comparison is done
        // with integers because of CUDA.  The return value of atomicCAS is
        // the original value at address_as_ull, so if it's equal to
        // "assumed", then nothing changed the value at the address while the
        // addition was being done.  If something changed the value at the
        // address "old" will not equal "assumed", and then retry the
        // addition.
        unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull;
        unsigned long long int assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull,
                            assumed,
                            __double_as_longlong(
                                v +  __longlong_as_double(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN
            // (since NaN != NaN)
        } while (assumed != old);
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
