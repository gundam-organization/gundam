#!/usr/bin/env python3
"""Run the shared migration/response contract through the actual Metal backend."""
import sys
from pathlib import Path
from importlib import import_module

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
fixture = import_module('220CpuDynamicBinning')

if __name__ == '__main__':
    if sys.platform != 'darwin':
        print('SKIP: Metal is only available on macOS')
    else:
        import GUNDAM
        if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
            print('SKIP: GUNDAM was built without backends')
        else:
            fixture.TOLERANCE = 2e-5
            fixture.main('MPS')
            print('SUCCESS: MPS migration, smoothing, cache invalidation and explicit readback agree')
