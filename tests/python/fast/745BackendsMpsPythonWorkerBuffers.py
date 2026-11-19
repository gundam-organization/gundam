#!/usr/bin/env python3
"""Check PythonWorker buffers, weights and gradients through the MPS backend."""
import sys

from shared.PythonWorkerBuffers import run


if __name__ == '__main__':
    sys.exit(run('MPS'))
