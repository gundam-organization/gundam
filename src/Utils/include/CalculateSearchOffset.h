#ifndef GUNDAM_CALCULATE_SEARCH_OFFSET_H
#define GUNDAM_CALCULATE_SEARCH_OFFSET_H

namespace {
  // Largest power of two <= count_, for the graph/spline binary search.
  // Integer operations keep this usable on MSVC, GCC, Clang and CUDA.
#ifdef __CUDACC__
  __host__ __device__
#endif
  inline unsigned int calculateSearchOffset(unsigned int count_){
#if defined(__CUDA_ARCH__)
    return count_ == 0 ? 0 : 1U << (31 - __clz(count_));
#elif defined(__GNUC__) || defined(__clang__)
    return count_ == 0 ? 0 : 1U << (31 - __builtin_clz(count_));
#else
    count_ |= count_ >> 1;
    count_ |= count_ >> 2;
    count_ |= count_ >> 4;
    count_ |= count_ >> 8;
    count_ |= count_ >> 16;
    return count_ - (count_ >> 1);
#endif
  }
}

#endif
