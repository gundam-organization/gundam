#!/usr/bin/env python3
"""Check near-minimum curvature and Hesse errors with both CPU paths."""
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
    import uproot
    config = {
        'likelihoodInterfaceConfig': {
            'jointProbabilityConfig': {'type': 'BarlowLLH_BANFF_OA2021', 'BBNoUpdateWeights': True},
            'backendManagerConfig': {'isEnabled': backend, 'type': 'CPU',
                                     'backendConfig': {'weightScheduling': 'Dynamic'}},
            'datasetList': [{'name': 'input', 'model': {
                'tree': 'events', 'filePathList': [str(path)], 'nominalWeightFormula': 'weight'}}],
            'propagatorConfig': {
                'sampleSetConfig': {'sampleList': [{
                    'name': 'sample', 'binning': {'binningDefinition': [{'name': 'X', 'edges': [0, 1]}]}}]},
                'parametersManagerConfig': {'parameterSetList': [{
                    'name': 'normalization', 'isEnabled': True, 'nominalStepSize': .1,
                    'parameterDefinitions': [{'name': 'scale', 'priorValue': 1., 'priorType': 'Flat',
                        'dialSetDefinitions': [{'dialType': 'Normalization'}]}]}]}}},
        'minimizerConfig': {'enablePostFitErrorFit': True, 'print_level': 0},
        'enablePca': False, 'scaleParStepWithChi2Response': False,
        'generateSamplePlots': False, 'generateOneSigmaPlots': False,
    }
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    li = engine.getLikelihoodInterface()
    li.setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    engine.initialize()
    parameter = li.getModelPropagator().getParametersManager().getParameterSetsList()[0].getParameterList()[0]
    content = li.getModelPropagator().getSampleSet().getSampleList()[0].getHistogram().getBinContentList()[0]
    mean, variance = content.sumWeights, content.sqrtSumSqWeights ** 2
    expected_curvature = 2 * mean * mean / (mean + variance)
    engine.evaluateLikelihood()
    nominal = li.getLastLikelihood()
    for step in (1e-5, 1e-6, 1e-7):
        values = []
        for value in (1. - step, 1. + step):
            parameter.setParameterValue(value)
            engine.evaluateLikelihood()
            values.append(li.getLastLikelihood())
        curvature = (sum(values) - 2 * nominal) / step ** 2
        assert math.isclose(curvature, expected_curvature, rel_tol=2e-6), (backend, step, curvature, expected_curvature)
    parameter.setParameterValue(1.)
    engine.evaluateLikelihood()
    output = Path.cwd() / f'230AsimovHesseStability-{backend}.root'
    app = GUNDAM.GundamApp('Asimov Hesse stability regression')
    app.openOutputFile(str(output))
    engine.setSaveDir(app, 'FitterEngine')
    engine.fit()
    with uproot.open(output) as saved:
        stats = saved['FitterEngine/postFit/Hesse/hesseStats'].arrays(library='np')
        assert stats['hesseStatusCode'][0] == 0, stats
        assert stats['covStatusCode'][0] == 3, stats
        histogram = saved['FitterEngine/postFit/Hesse/errors/normalization/values/postFitErrors_TH1D']
        error = math.sqrt(float(histogram.variances()[0]))
    expected_error = math.sqrt(2 / expected_curvature)
    assert math.isclose(error, expected_error, rel_tol=2e-4), (backend, error, expected_error)
    return error


def main():
    import GUNDAM
    import numpy as np
    import uproot
    GUNDAM.setNumberOfThreads(4)
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / '230AsimovHesseStability-input.root'
    with uproot.recreate(path) as saved:
        tree = saved.mktree('events', {'X': 'float64', 'weight': 'float64'})
        tree.extend({'X': np.full(4001, .5), 'weight': .125 + .01 * (np.arange(4001) % 7)})
    reference = evaluate(path, False)
    backend = evaluate(path, True)
    assert math.isclose(reference, backend, rel_tol=2e-4), (reference, backend)
    print('SUCCESS: CPU and backend Hesse uncertainties match the analytic Asimov curvature.')


if __name__ == '__main__':
    main()
