#!/usr/bin/env python3
"""Strict observable composition through BackendManager; CUDA opt-in in a GPU job."""
import math
import os
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
fixture = import_module('219VariableDials')
capture = import_module('220CpuDynamicBinning').capture


def main():
    import GUNDAM

    if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        if os.environ.get('GUNDAM_TEST_CUDA') == '1':
            raise RuntimeError('CUDA validation requested but backends are unavailable')
        print('SKIP: built without backends')
        return 0
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(2)
    path = Path.cwd() / '230BackendObservableResponses.root'
    fixture.write_input(path)

    def run(backend, mode=None):
        engine = fixture.make_engine(path, backend=backend, strict_only=True, backend_mode=mode)
        li = engine.getLikelihoodInterface()
        propagator = li.getModelPropagator()
        parameters = {p.getName(): p for p in
                      propagator.getParametersManager().getParameterSetsList()[0].getParameterList()}
        manager = engine.getBackendsManager() if backend else None
        if manager:
            assert manager.hasBackend() and manager.getType() == backend
            manager.setEnableAutoMaterialize(False)
        rows = []
        for multiply, add, override, weight in [(1., .5, 1.25, 2.), (2., -.25, -.5, 1.5),
                                                (1., .75, 1., 3.), (1., .5, 1.25, 2.)]:
            for name, value in [('multiply', multiply), ('add', add), ('override', override), ('weight', weight)]:
                parameters[name].setParameterValue(value)
            for _ in range(2):
                engine.evaluateLikelihood()
                if manager:
                    # Explicit state readback must restore variables, bins and host pointer lists.
                    manager.materialize('EventState')
                    manager.materialize('EventWeights')
                    manager.materialize('Histograms')
                values = [(override if i % 2 else raw) * multiply * .75 + add - .25
                          for i, raw in enumerate(fixture.RAW)]
                fixture.check(propagator, values, add, weight)
                fixture.check(li.getDataPropagator(), [.75 * (1.25 if i % 2 else x) + .25
                                                     for i, x in enumerate(fixture.RAW)], .5, 2.)
                rows.append(capture(engine))
        return rows

    reference = run(None)
    backends = [('CPU', None)]
    if os.environ.get('GUNDAM_TEST_CUDA') == '1':
        backends.extend(('CUDA', mode) for mode in ('cachedResponses', 'recomputeResponses'))
    for backend, mode in backends:
        actual = run(backend, mode)
        for point, (row, expected) in enumerate(zip(actual, reference)):
            assert len(row) == len(expected)
            for output, (a, b) in enumerate(zip(row, expected)):
                assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (backend, mode, point, output, a, b)
    print('SUCCESS: strict observable composition, migrations, materialization and Asimov copies agree')
    return 0


if __name__ == '__main__':
    sys.exit(main())
