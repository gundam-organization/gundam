#!/usr/bin/env python3
"""Keep Python migration responses and binning cached until their inputs change."""
import json
import math
import os
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)


def main():
    import GUNDAM
    import numpy as np
    import uproot

    if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        if os.environ.get('GUNDAM_TEST_CUDA') == '1':
            raise RuntimeError('CUDA validation requested but backends are unavailable')
        print('SKIP: built without backends')
        return
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(2)
    root = Path.cwd() / '222VariableResponseDependencies.root'
    with uproot.recreate(root) as output:
        tree = output.mktree('events', {'X': 'float64'})
        tree.extend({'X': np.array([.25, .75, 1.25, 1.75])})

    def run(backend, mode=None, strict=False):
        calls = Path.cwd() / f'222VariableResponseDependencies_{backend}_{mode}_{strict}.calls'
        calls.write_text('')
        script = calls.with_suffix('.py')
        script.write_text(f'''from pathlib import Path
def run(command):
    if command['command'] == 'evaluate':
        with Path({str(calls)!r}).open('a') as output:
            output.write('evaluate\\n')
        command['weights'][:] = command['inputs']['X'] + command['parameters']['shift']
    return {{'status': 'ok'}}
''')
        sample = {'name': 'sample', 'binning': {'binningDefinition': [
            {'name': 'X', 'edges': [0., 1., 2.]}]},
            'binningResponse': {'isEnabled': True, 'variables': [{'name': 'X', 'width': .2}]}}
        if strict:
            sample.pop('binningResponse')
        migration = {'dialType': 'ExternalWeight', 'applyToVariable': 'X',
                     'variableResponseMode': 'Override', 'dialInputList': [{'name': 'shift'}],
                     'options': {'type': 'PythonWorker', 'inputEventVarList': ['X'],
                                 'workerConfig': {'pythonExecutable': sys.executable, 'evalScript': str(script)}}}
        li = {'jointProbabilityConfig': {'type': 'LeastSquares', 'lsqPoissonianApproximation': True},
              'dataSetList': [{'name': 'inputs', 'model': {'tree': 'events', 'filePathList': [str(root)]}}],
              'propagatorConfig': {'sampleSetConfig': {'sampleList': [sample]},
                                  'parametersManagerConfig': {'parameterSetList': [{
                                      'name': 'parameters', 'isEnabled': True, 'nominalStepSize': .1,
                                      'parameterDefinitions': [
                                          {'name': 'shift', 'priorValue': 0., 'priorType': 'Flat',
                                           'dialSetDefinitions': [migration]},
                                          {'name': 'weight', 'priorValue': 1., 'priorType': 'Flat',
                                           'dialSetDefinitions': [{'dialType': 'Normalization'}]}]}]}}}
        if backend != 'standard':
            li['backendManagerConfig'] = {'isEnabled': True, 'useBackend': backend,
                'options': [{'name': backend, **({'propagationMode': mode} if mode else {})}]}
        config = {'likelihoodInterfaceConfig': li, 'enablePca': False,
                  'scaleParStepWithChi2Response': False, 'generateSamplePlots': False,
                  'generateOneSigmaPlots': False}
        engine = GUNDAM.FitterEngine()
        engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
        engine.initialize()
        propagator = engine.getLikelihoodInterface().getModelPropagator()
        parameters = propagator.getParametersManager().getParameterSetsList()[0].getParameterList()
        manager = engine.getBackendsManager() if backend != 'standard' else None
        if manager:
            assert manager.hasBackend()
            manager.setEnableAutoMaterialize(False)
        results = []

        def evaluate(expected_calls, expect_rebin=False):
            before = len(calls.read_text().splitlines())
            engine.evaluateLikelihood()
            assert len(calls.read_text().splitlines()) - before == expected_calls, backend
            if manager:
                timing = manager.getLastTimingSummary()
                assert (timing.rebinnedEventCount > 0) == expect_rebin, backend
                if backend in ('MPS', 'CUDA'):
                    assert timing.externalWeightUploadBlocks == expected_calls
                manager.materialize('Histograms')
            histogram = propagator.getSampleSet().getSampleList()[0].getHistogram()
            results.append([v for b in histogram.getBinContentList() for v in (b.sumWeights, b.sqrtSumSqWeights)]
                           + [histogram.getOutsideBinContent().sumWeights])

        evaluate(0)
        parameters[1].setParameterValue(2.)
        evaluate(0)
        parameters[0].setParameterValue(.5)
        evaluate(1, True)
        parameters[0].setParameterValue(.5)
        evaluate(0)
        # Explicit invalidation still reevaluates the producer at an identical point.
        for entry in propagator.getEventDialCache().getCache():
            for response in entry.getDialResponseCacheList():
                response.getDialInterface().getInputBuffer().invalidateBuffers()
        evaluate(1)
        evaluate(0)
        return results

    reference = run('standard')
    for backend in ['CPU'] + (['MPS'] if sys.platform == 'darwin' else []):
        actual = run(backend)
        tolerance = 2e-5 if backend == 'MPS' else 1e-10
        for row, expected in zip(actual, reference):
            assert all(math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)
                       for a, b in zip(row, expected)), (backend, row, expected)
    if os.environ.get('GUNDAM_TEST_CUDA') == '1':
        for strict in (True, False):
            reference = run('standard', strict=strict)
            for mode in ('cachedResponses', 'recomputeResponses'):
                actual = run('CUDA', mode=mode, strict=strict)
                for row, expected in zip(actual, reference):
                    assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
                               for a, b in zip(row, expected)), (mode, strict, row, expected)
    print('SUCCESS: observable producer calls, cached binning and explicit invalidation agree')


if __name__ == '__main__':
    main()
