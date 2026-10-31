#!/usr/bin/env python3
"""Check finite differences away from zero likelihood and a MPS toy minimum."""
import json
import math
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)


def evaluate(path, backend):
    import GUNDAM
    config = {
        'likelihoodInterfaceConfig': {
            'jointProbabilityConfig': {'type': 'PoissonLLH'},
            'backendManagerConfig': {'isEnabled': True, 'type': backend},
            'datasetList': [{'name': 'input', 'model': {
                'tree': 'events', 'filePathList': [str(path)], 'nominalWeightFormula': 'weight'}}],
            'propagatorConfig': {
                'sampleSetConfig': {'sampleList': [{
                    'name': 'sample', 'binning': {'binningDefinition': [{'name': 'X', 'edges': [0, 1, 2]}]}}]},
                'parametersManagerConfig': {'parameterSetList': [{
                    'name': 'normalization', 'isEnabled': True, 'nominalStepSize': .1,
                    'parameterDefinitions': [{'name': 'scale', 'priorValue': 1., 'priorType': 'Flat',
                        'dialSetDefinitions': [{'dialType': 'Normalization'}]}]}]}}},
        'minimizerConfig': {'enablePostFitErrorFit': False, 'print_level': 0},
        'enablePca': False, 'scaleParStepWithChi2Response': False,
        'generateSamplePlots': False, 'generateOneSigmaPlots': False,
    }

    GUNDAM.FitterEngine.setRandomSeed(1234)
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    li = engine.getLikelihoodInterface()
    li.setDataType(GUNDAM.LikelihoodInterface.DataType.Toy)
    engine.initialize()
    parameter = li.getModelPropagator().getParametersManager().getParameterSetsList()[0].getParameterList()[0]
    def contents(propagator):
        return [c.sumWeights for c in propagator.getSampleSet().getSampleList()[0].getHistogram().getBinContentList()]
    mean = sum(contents(li.getModelPropagator()))
    observed = sum(contents(li.getDataPropagator()))
    point = 1.1
    def objective(value):
        parameter.setParameterValue(value)
        engine.evaluateLikelihood()
        return li.getLastLikelihood()
    nominal = objective(point)
    assert nominal > .1, nominal
    for step in (1e-3, 1e-4):
        minus, plus = objective(point - step), objective(point + step)
        gradient = (plus - minus) / (2 * step)
        curvature = (plus + minus - 2 * nominal) / step ** 2
        expected_gradient = 2 * (mean - observed / point)
        expected_curvature = 2 * observed / point ** 2
        assert math.isclose(gradient, expected_gradient, rel_tol=1e-5), (backend, step, gradient, expected_gradient)
        assert math.isclose(curvature, expected_curvature, rel_tol=1e-4), (backend, step, curvature, expected_curvature)
    parameter.setParameterValue(1.)
    engine.evaluateLikelihood()
    engine.fit()
    assert engine.getMinimizer().hasConverged(), backend
    fitted = parameter.getParameterValue()
    assert math.isclose(fitted, observed / mean, rel_tol=1e-4), (backend, fitted, observed / mean)
    return fitted


def main():
    if sys.platform != 'darwin':
        print('SKIP: Metal is only available on macOS')
        return
    import GUNDAM
    if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        print('SKIP: GUNDAM was built without backends')
        return
    import numpy as np
    import uproot
    GUNDAM.setNumberOfThreads(4)
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / '236MpsToyPrecision-input.root'
    with uproot.recreate(path) as saved:
        tree = saved.mktree('events', {'X': 'float64', 'weight': 'float64'})
        tree.extend({'X': .5 + (np.arange(4001) % 2), 'weight': .125 + .01 * (np.arange(4001) % 7)})
    cpu, mps = evaluate(path, 'CPU'), evaluate(path, 'MPS')
    assert math.isclose(cpu, mps, rel_tol=1e-5), (cpu, mps)
    print('SUCCESS: MPS toy gradients, curvature and fitted normalization match analytic values and CPU.')


if __name__ == '__main__':
    main()
