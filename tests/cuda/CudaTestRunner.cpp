#include "CudaTestDevice.h"

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <cerrno>
#include <cstring>

int main(int argc_, char** argv_) {
  if( argc_ < 2 ){
    std::cerr << "Usage: gundamCudaTestRunner <test-executable> [arguments...]\n";
    return 1;
  }
  const int status = checkCudaTestDevice();
  if( status != 0 ){ return status; }
  // Preserve arguments and the child's exit status without a shell. Once a
  // GPU is available, kernel and numerical failures must remain failures.
#ifdef _WIN32
  const auto result = _spawnv(_P_WAIT, argv_[1], argv_ + 1);
  if( result != -1 ){ return static_cast<int>(result); }
#else
  execv(argv_[1], argv_ + 1);
#endif
  std::cerr << "Cannot launch CUDA test: " << std::strerror(errno) << '\n';
  return 1;
}
