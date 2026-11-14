#ifndef GUNDAM_WORKER_BUFFER_H
#define GUNDAM_WORKER_BUFFER_H

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

// Backend-independent ownership and transport contract for PythonWorker buffers.
// Implementations live in the backend; DialDictionary never includes Metal/CUDA.
struct WorkerBufferDescriptor {
  std::string transport, path, libraryPath;
  std::size_t count{0}, byteCount{0};
};

class WorkerBuffer {
public:
  virtual ~WorkerBuffer() = default;
  virtual WorkerBufferDescriptor describe() const = 0;
  virtual void writeValues(const double* values_, std::size_t count_) = 0;
  virtual void readValues(double* values_, std::size_t count_) const = 0;
  virtual void* nativeHandle() const = 0;
};

using WorkerBufferFactory = std::function<std::shared_ptr<WorkerBuffer>(std::size_t)>;

#endif
