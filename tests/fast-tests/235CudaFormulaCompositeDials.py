#!/usr/bin/env python3
"""Opt-in CUDA comparison against ROOT: formulas, composites and event migration."""
from functools import partial
from importlib import import_module
from pathlib import Path
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    if os.environ.get('GUNDAM_TEST_CUDA') != '1':
        print('SKIP: set GUNDAM_TEST_CUDA=1 inside a CUDA GPU allocation.')
        return
    import_module('005PythonSetup-utils').requireVenv(__file__)
    import GUNDAM
    assert hasattr(GUNDAM.FitterEngine, 'getBackendsManager'), 'Rebuild with WITH_BACKENDS=ON'
    composite = import_module('231CpuCompositeDials')
    formula = import_module('232CpuFormulaDials')
    originalComposite, originalFormula = composite.compare, formula.compare
    for mode in ('cachedResponses', 'recomputeResponses'):
        for layout in ('eventMajor', 'warpInterleaved'):
            options = dict(backendType='CUDA', backendConfig={
                'propagationMode': mode, 'weightLayout': layout})
            composite.compare = partial(originalComposite, exact=False, **options)
            formula.compare = partial(originalFormula, **options)
            composite.main()
            formula.main()
    print('SUCCESS: CUDA formulas and composites match ROOT in both modes and layouts.')


if __name__ == '__main__':
    main()
