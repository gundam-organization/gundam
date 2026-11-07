#include <cuda_runtime.h>

#include <array>
#include <iostream>

namespace {
  bool checkCuda(cudaError_t status_, const char* operation_) {
    if(status_ == cudaSuccess){ return true; }
    std::cerr << operation_ << ": " << cudaGetErrorString(status_) << std::endl;
    return false;
  }

  __global__ void fillValues(double* values_, int size_) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size_){ values_[index] = 2. * index + 1.; }
  }
}

int main() {
  int deviceCount = 0;
  if(not checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount")){ return 1; }
  if(deviceCount == 0){
    std::cerr << "No CUDA device visible. Run this test inside a GPU allocation." << std::endl;
    return 1;
  }
  cudaDeviceProp properties{};
  if(not checkCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties")){ return 1; }
  std::cout << "Device 0: " << properties.name << " (sm_"
            << properties.major << properties.minor << ")" << std::endl;

  // A partial final block also exercises the kernel bounds check.
  std::array<double, 257> values{};
  double* deviceValues = nullptr;
  if(not checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceValues), sizeof(values)), "cudaMalloc")){ return 1; }
  fillValues<<<3, 128>>>(deviceValues, static_cast<int>(values.size()));
  const bool success = checkCuda(cudaGetLastError(), "kernel launch")
      and checkCuda(cudaDeviceSynchronize(), "kernel execution")
      and checkCuda(cudaMemcpy(values.data(), deviceValues, sizeof(values), cudaMemcpyDeviceToHost), "cudaMemcpy");
  const bool freed = checkCuda(cudaFree(deviceValues), "cudaFree");
  if(not success or not freed){ return 1; }
  for(std::size_t i = 0; i < values.size(); ++i){
    if(values[i] != 2. * i + 1.){
      std::cerr << "Incorrect result at index " << i << ": " << values[i] << std::endl;
      return 1;
    }
  }
  std::cout << "CUDA allocation, FP64 kernel, synchronization and readback passed." << std::endl;
  return 0;
}
