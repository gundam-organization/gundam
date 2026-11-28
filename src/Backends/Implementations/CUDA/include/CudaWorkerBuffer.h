#ifndef GUNDAM_CUDA_WORKER_BUFFER_H
#define GUNDAM_CUDA_WORKER_BUFFER_H

#include "WorkerBuffer.h"

namespace Backends {
  std::shared_ptr<WorkerBuffer> makeCudaWorkerBuffer(std::size_t count_, int deviceIndex_);
}

#endif
