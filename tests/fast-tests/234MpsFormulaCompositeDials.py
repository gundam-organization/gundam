#!/usr/bin/env python3
"""Metal formula/composite dials match ROOT, including event inputs and migration."""
from functools import partial
from importlib import import_module
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)


def main():
    if sys.platform != 'darwin':
        print('SKIP: Metal requires macOS.')
        return
    import GUNDAM
    if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        print('SKIP: backend support was not compiled.')
        return
    composite = import_module('231CpuCompositeDials')
    formula = import_module('232CpuFormulaDials')
    originalComposite = composite.compare
    originalFormula = formula.compare
    for mode in ('cachedResponses', 'recomputeResponses'):
        options = dict(backendType='MPS', backendConfig={'propagationMode': mode})
        composite.compare = partial(originalComposite, exact=False, **options)
        formula.compare = partial(originalFormula, **options)
        composite.main()
        formula.main()
    print('SUCCESS: MPS formulas and composites match ROOT in both execution modes.')


if __name__ == '__main__':
    main()
