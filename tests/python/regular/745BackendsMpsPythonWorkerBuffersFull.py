#!/usr/bin/env python3
"""Full PythonWorker transport coverage complementary to the fast buffer check."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fast'))

from shared.PythonWorkerBuffers import run


if __name__ == '__main__':
    sys.exit(run('MPS', full=True))
