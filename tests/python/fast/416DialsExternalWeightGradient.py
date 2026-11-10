#!/usr/bin/env python3
"""External workers supply dense event/bin gradients and acknowledge their generation."""
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

WORKER = '''
import numpy as np
mode = 'ok'
def configure(arguments):
    global mode
    mode = arguments[0]
def run(command):
    if command['command'] == 'initialize':
        return {'status': 'ok', 'supportsGradient': mode != 'unsupported'}
    if command['command'] == 'shutdown':
        return {'status': 'ok'}
    a = command['parameters']['a']
    b = command['parameters']['b']
    x = command['inputs']['X']
    command['weights'][:] = 1 + a*a*(1+x) + b*x
    result = {'status': 'ok'}
    if command['withGradient']:
        gradient = command['gradients']
        assert gradient.shape == (command['nBins'], 2)
        # The declared input order is b, a, not ROOT/parameter-definition order.
        gradient[:, 0] = x
        if mode != 'partial':
            gradient[:, 1] = 2*a*(1+x)
        if mode == 'nonfinite':
            gradient[0, 0] = np.inf
        if mode != 'missingAck':
            result['gradientGeneration'] = command['generation'] + (mode == 'staleAck')
    return result
'''


def config(path, script, binned, mode):
    external = {
        'dialType': 'ExternalWeight',
        'dialInputList': [{'name': 'b'}, {'name': 'a'}],
        'options': {
            'type': 'PythonWorker', 'useBinnedWeights': binned, 'inputEventVarList': ['X'],
            'workerConfig': {'pythonExecutable': sys.executable, 'evalScript': str(script),
                             'scriptArgs': [mode]}}}
    if binned:
        external['options']['binning'] = {'binningDefinition': [{'name': 'X', 'edges': [0, .5, 1]}]}
    return {
        'enablePca': False, 'scaleParStepWithChi2Response': False,
        'generateSamplePlots': False, 'generateOneSigmaPlots': False,
        'minimizerConfig': {'useAnalyticGradient': True, 'useNormalizedFitSpace': False,
                            'print_level': 0},
        'likelihoodInterfaceConfig': {
            'jointProbabilityConfig': {'type': 'BarlowBeestonLiteGaussian'},
            'datasetList': [{'name': 'input', 'model': {'tree': 'events', 'filePathList': [str(path)]}}],
            'propagatorConfig': {
                'sampleSetConfig': {'sampleList': [{'name': 'sample', 'binning': {
                    'binningDefinition': [{'name': 'X', 'edges': [0, .5, 1]}]}}]},
                'parametersManagerConfig': {'parameterSetList': [{
                    'name': 'parameters', 'isEnabled': True, 'nominalStepSize': .05,
                    'parameterDefinitions': [
                        {'name': 'a', 'priorValue': .4, 'priorType': 'Flat'},
                        {'name': 'b', 'priorValue': .2, 'priorType': 'Flat'}],
                    'dialSetDefinitions': [external]}]}}}}


def build(configuration):
    import GUNDAM
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(configuration))))
    engine.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    engine.initialize()
    return engine


def main():
    import GUNDAM
    import numpy as np
    import uproot

    path = Path('416DialsExternalWeightGradient.root').resolve()
    script = Path('416DialsExternalWeightGradient_worker.py').resolve()
    script.write_text(WORKER)
    with uproot.recreate(path) as output:
        tree = output.mktree('events', {'X': 'float64'})
        tree.extend({'X': np.array([.1, .2, .6, .8])})
    GUNDAM.setLightOutputMode(True)
    def with_backend(configuration, backend):
        if backend:
            configuration['likelihoodInterfaceConfig']['backendManagerConfig'] = {
                'isEnabled': True, 'useBackend': backend}
        return configuration

    for backend in (None, 'CPU'):
        for threads in (1, 4):
            GUNDAM.setNumberOfThreads(threads)
            for binned in (False, True):
                for smooth in (False, True):
                    configuration = config(path, script, binned, 'ok')
                    if smooth:
                        dial = configuration['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['dialSetDefinitions'][0]
                        dial.update(dialResponseLimits=[1.1, 1.5], enableResponseSmoothing=True,
                                    dialResponseSmoothingWidth=.15)
                    engine = build(with_backend(configuration, backend))
                    minimizer = engine.getMinimizer()
                    for point in ([.55, .3], [.55, .3], [.2, -.1]):
                        # Initialization has already evaluated weights alone. The first
                        # gradient call must recompute at this point with derivatives.
                        analytic = minimizer.evalFitGradientAt(point)
                        for index, actual in enumerate(analytic):
                            up, down = list(point), list(point)
                            up[index] += 1e-5
                            down[index] -= 1e-5
                            numeric = (minimizer.evalFit(up) - minimizer.evalFit(down)) / 2e-5
                            assert math.isclose(actual, numeric, rel_tol=3e-5, abs_tol=3e-6), (actual, numeric)
                    del minimizer, engine
        for mode in ('unsupported', 'partial', 'nonfinite', 'missingAck', 'staleAck'):
            for binned in (False, True):
                try:
                    engine = build(with_backend(config(path, script, binned, mode), backend))
                    engine.getMinimizer().evalFitGradientAt([.55, .3])
                except RuntimeError as error:
                    expected = {
                        'unsupported': 'unsupported',
                        'partial': 'missing/non-finite gradient',
                        'nonfinite': 'missing/non-finite gradient',
                        'missingAck': 'gradientGeneration',
                        'staleAck': 'gradientGeneration',
                    }[mode]
                    assert expected.lower() in str(error).lower(), (mode, str(error))
                else:
                    raise AssertionError('Invalid external gradient accepted: ' + mode)
    print('SUCCESS: external event/bin gradients match finite differences and invalid workers fail.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
