#ifndef GUNDAM_CUDA_TEST_DEVICE_H
#define GUNDAM_CUDA_TEST_DEVICE_H

#include <cuda_runtime_api.h>

#include <iostream>

// CTest interprets 77 as skipped. Compilation only requires the toolkit;
// device tests require a visible GPU and an initialized, compatible driver.
inline int checkCudaTestDevice() {
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  if( status == cudaErrorNoDevice or status == cudaErrorInsufficientDriver
      or status == cudaErrorInitializationError or (status == cudaSuccess and count == 0) ){
    std::cout << "CUDA test skipped: "
              << (status == cudaSuccess ? "no visible CUDA GPU" : cudaGetErrorString(status))
              << ". Run in a GPU allocation with a compatible driver.\n";
    return 77;
  }
  if( status != cudaSuccess ){
    std::cerr << "CUDA device detection failed: " << cudaGetErrorString(status) << '\n';
    return 1;
  }
  return 0;
}

#endif // GUNDAM_CUDA_TEST_DEVICE_H
