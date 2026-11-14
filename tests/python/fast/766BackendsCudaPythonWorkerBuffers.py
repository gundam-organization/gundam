#!/usr/bin/env python3
"""Check PythonWorker buffers, weights and gradients through CUDA IPC and DLPack."""
import sys

from shared.PythonWorkerBuffers import run


if __name__ == '__main__':
    for mode in ('cachedResponses', 'recomputeResponses'):
        status = run('CUDA', propagationMode=mode)
        if status:
            sys.exit(status)
