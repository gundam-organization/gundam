#!/usr/bin/env python3
"""Check MPS Hesse uncertainties against analytic Asimov curvature."""
import json
import math
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)


def evaluate(path, precision=None):
    import GUNDAM
    import uproot
    config = {
        'likelihoodInterfaceConfig': {
            'jointProbabilityConfig': {'type': 'BarlowLLH_BANFF_OA2021', 'BBNoUpdateWeights': True},
            'backendManagerConfig': {'isEnabled': True, 'type': 'MPS'},
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
    if precision is not None:
        config['minimizerConfig']['functionPrecision'] = precision
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    li = engine.getLikelihoodInterface()
    li.setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    output = Path.cwd() / f'231MpsHessePrecision-{precision}.root'
    app = GUNDAM.GundamApp('MPS Hesse precision regression')
    app.openOutputFile(str(output))
    engine.setSaveDir(app, 'FitterEngine')
    engine.initialize()
    expected_precision = precision if precision is not None else 2 ** -23
    assert engine.getMinimizer().getFunctionPrecision() == expected_precision
    parameter = li.getModelPropagator().getParametersManager().getParameterSetsList()[0].getParameterList()[0]
    content = li.getModelPropagator().getSampleSet().getSampleList()[0].getHistogram().getBinContentList()[0]
    mean, variance = content.sumWeights, content.sqrtSumSqWeights ** 2
    expected_curvature = 2 * mean * mean / (mean + variance)
    engine.evaluateLikelihood()
    nominal = li.getLastLikelihood()
    for step in (1e-2, 1e-3):
        values = []
        for value in (1. - step, 1. + step):
            parameter.setParameterValue(value)
            engine.evaluateLikelihood()
            values.append(li.getLastLikelihood())
        curvature = (sum(values) - 2 * nominal) / step ** 2
        assert math.isclose(curvature, expected_curvature, rel_tol=5e-3), (precision, step, curvature, expected_curvature)
    parameter.setParameterValue(1.)
    engine.evaluateLikelihood()
    engine.fit()
    with uproot.open(output) as saved:
        saved_precision = saved['FitterEngine/fit/minimizer/functionPrecision_TNamed'].member('fTitle')
        assert float(saved_precision) == expected_precision, saved_precision
        stats = saved['FitterEngine/postFit/Hesse/hesseStats'].arrays(library='np')
        # ROOT's combined status stores the Hesse code in the hundreds digit.
        assert (stats['hesseStatusCode'][0] // 100) % 10 == 0, stats
        assert stats['covStatusCode'][0] == 3, stats
        histogram = saved['FitterEngine/postFit/Hesse/errors/normalization/values/postFitErrors_TH1D']
        error = math.sqrt(float(histogram.variances()[0]))
    expected_error = math.sqrt(2 / expected_curvature)
    assert math.isclose(error, expected_error, rel_tol=5e-3), (precision, error, expected_error)
    return error


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
    path = Path.cwd() / '231MpsHessePrecision-input.root'
    with uproot.recreate(path) as saved:
        tree = saved.mktree('events', {'X': 'float64', 'weight': 'float64'})
        tree.extend({'X': np.full(4001, .5), 'weight': .125 + .01 * (np.arange(4001) % 7)})
    evaluate(path)
    evaluate(path, 1e-6)
    for invalid in (0., -1., float('inf'), float('nan')):
        engine = GUNDAM.FitterEngine()
        config = {'minimizerConfig': {'functionPrecision': invalid}}
        try:
            engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
        except RuntimeError as error:
            assert 'functionPrecision' in str(error), str(error)
        else:
            raise AssertionError(f'Accepted invalid function precision: {invalid}')
    print('SUCCESS: MPS Hesse matches analytic curvature; precision overrides and validation work.')


if __name__ == '__main__':
    main()
