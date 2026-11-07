#!/usr/bin/env python3
"""Metal formula/composite dials match ROOT, including event inputs and migration."""
from functools import partial
from importlib import import_module
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(full=False):
    if sys.platform != 'darwin':
        print('SKIP: Metal requires macOS.')
        return
    import GUNDAM
    if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        print('SKIP: backend support was not compiled.')
        return
    composite = import_module('723BackendsCpuCompositeDials')
    formula = import_module('725BackendsCpuFormulaDials')
    originalComposite = composite.compare
    originalFormula = formula.compare
    for mode in ('cachedResponses', 'recomputeResponses'):
        options = dict(backendType='MPS', backendConfig={'propagationMode': mode})
        if full:
            composite.compare = partial(originalComposite, exact=False, **options)
            formula.compare = partial(originalFormula, **options)
            composite.main()
            formula.main()
        else:
            GUNDAM.setLightOutputMode(True)
            GUNDAM.setNumberOfThreads(1)
            path = Path.cwd() / '743BackendsMpsFormulaCompositeDials.root'
            if mode == 'cachedResponses':
                composite.helpers.write_input(path)
            # A formula with parameter and event inputs feeds a composite
            # that changes the observable, exercising bin migration too.
            leaf = composite.helpers.formula('shift', '[arg0]+[arg1]',
                                             {'eventVariable': 'X'}, {'parameterName': 'p'})
            dial = composite.helpers.composite([leaf], 'shift', applyToVariable='X',
                                                variableResponseMode='Override')
            originalComposite(path, dial, exact=False, **options)
    print('SUCCESS: MPS formulas and composites match ROOT in both execution modes.')


if __name__ == '__main__':
    main()
