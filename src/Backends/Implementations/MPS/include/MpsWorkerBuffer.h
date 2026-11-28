#ifndef GUNDAM_MPS_WORKER_BUFFER_H
#define GUNDAM_MPS_WORKER_BUFFER_H

#include "WorkerBuffer.h"

namespace Backends {
  std::shared_ptr<WorkerBuffer> makeMpsWorkerBuffer(std::size_t count_);
  // Copies/converts weights on GPU to the backend's float or float-pair layout.
  void copyMpsWorkerWeights(const WorkerBuffer& source_, void* destination_,
                            std::size_t offset_, std::size_t count_, std::size_t scalarBytes_);
}

#endif
