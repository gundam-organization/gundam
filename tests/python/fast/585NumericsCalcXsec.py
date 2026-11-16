#!/usr/bin/env python3
"""Exact CalcXsec checks: weighted yields, bin volumes, toys and ROOT outputs.

The JSON-compatible configurations below are also saved beside each result.
All checks launch gundamCalcXsec; no Python reimplementation replaces the app.
"""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helpers'))
from CalcXsec import calculate, read_output, write_config, write_covariance, write_events
import numpy as np


def configurations(events, covariance):
    # Two normalization parameters act on different x bins, with rho = -0.5.
    # category selects the sample; y has width 3; the third x bin is empty.
    fitter = {'fitterEngineConfig': {
        'likelihoodInterfaceConfig': {
            'jointProbabilityConfig': {'type': 'PoissonLLH', 'ignoreBinsWithZeroPredictionAtPrior': True},
            'datasetList': [{'name': 'signal', 'model': {
                'tree': 'events', 'filePathList': [str(events)],
                'nominalWeightFormula': 'weight',
            }}],
            'propagatorConfig': {
                'parametersManagerConfig': {'parameterSetList': [{
                    'name': 'response', 'isEnabled': True, 'parameterDefinitionFilePath': str(covariance),
                    'covarianceMatrix': 'covariance', 'nominalStepSize': 0.01,
                    'parameterDefinitions': [
                        {'name': 'low', 'parameterIndex': 0, 'priorValue': 1.,
                         'dialSetDefinitions': [{'dialType': 'Normalization', 'applyCondition': 'x < 2'}]},
                        {'name': 'high', 'parameterIndex': 1, 'priorValue': 1.,
                         'dialSetDefinitions': [{'dialType': 'Normalization', 'applyCondition': 'x >= 2'}]},
                    ],
                }]},
                # These fit samples must be replaced by the extraction samples.
                'sampleSetConfig': {'sampleList': [{
                    'name': 'fitOnly', 'binning': {'binningDefinition': [{'name': 'x', 'edges': [0, 8]}]},
                }]},
            },
        },
    }}
    xsec = {
        'xsecCalcConfig': {'enableStatThrowInToys': False, 'enableEventMcThrow': False},
        'fitterEngineConfig': {'likelihoodInterfaceConfig': {'propagatorConfig': {
            'sampleSetConfig': {'sampleList': [{
                'name': name, 'selectionCutStr': f'category == {category}',
                'binning': {'binningDefinition': [
                    {'name': 'x', 'edges': [0, 2, 6, 8]},
                    {'name': 'y', 'edges': [0, 3]},
                ]},
                'normaliseParameterList': [{'name': 'targets', 'meanValue': 10.}],
                'yAxis': 'dSigma',
            } for category, name in enumerate(('signalA', 'signalB'))]},
        }}},
    }
    return fitter, xsec


def check_linear(toys, nominal, factors=(2., 2., 0., 2., 2., 0.)):
    # Six sample bins followed by the two parameter leaves.
    assert toys.shape[1] == 8
    expected = toys[:, [6, 7, 7, 6, 7, 7]] * np.array(factors)
    np.testing.assert_allclose(toys[:, :6], expected, atol=1e-10)
    np.testing.assert_allclose(nominal[:6], factors, atol=1e-10)
    np.testing.assert_allclose(nominal[6:], [1., 1.], atol=1e-10)


def main():
    fitter, xsec = configurations(write_events(), write_covariance())
    fitter_path = write_config('fitter', fitter)
    result = calculate('linear', fitter_path, xsec)
    toys, nominal, covariance = read_output(result, 32)
    check_linear(toys, nominal)

    repeated = read_output(calculate('repeat', fitter_path, xsec), 32)[0]
    np.testing.assert_array_equal(toys, repeated)
    different = read_output(calculate('otherSeed', fitter_path, xsec, seed=6789), 32)[0]
    assert not np.array_equal(toys[:, 6:], different[:, 6:])
    threaded = read_output(calculate('threads', fitter_path, xsec, threads=2), 32)[0]
    np.testing.assert_allclose(toys, threaded, atol=1e-10)

    # Excluding y from the differential volume multiplies sigma by three.
    excluded = deepcopy(xsec)
    samples = excluded['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig']['sampleSetConfig']['sampleList']
    for sample in samples:
        sample['normaliseParameterList'] += [{'name': 'integrateY', 'disabledBinDim': 'y'}]
    check_linear(*read_output(calculate('excludedDimension', fitter_path, excluded), 32)[:2],
                 factors=(6., 6., 0., 6., 6., 0.))

    # Constant normalizers multiply. Disabled entries have no numerical effect.
    normalized = deepcopy(xsec)
    for sample in normalized['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig']['sampleSetConfig']['sampleList']:
        sample['normaliseParameterList'] += [
            {'name': 'efficiency', 'meanValue': 0.5},
            {'name': 'disabled', 'isEnabled': False, 'meanValue': 100.},
        ]
    check_linear(*read_output(calculate('normalizers', fitter_path, normalized), 32)[:2],
                 factors=(4., 4., 0., 4., 4., 0.))

    for mode in ('cli', 'config'):
        central = deepcopy(xsec)
        options = ('--use-bf-as-xsec',) if mode == 'cli' else ()
        if mode == 'config':
            central['useBestFitAsCentralValue'] = True
        other_toys, _, other_covariance = read_output(
            calculate('bestFit' + mode, fitter_path, central, options=options), 32, best_fit=True)
        np.testing.assert_array_equal(other_toys, toys)
        np.testing.assert_array_equal(other_covariance, covariance)
    assert not np.allclose(toys[:, :6].mean(axis=0), nominal[:6])

    # CLI overrides must reach the calculator configuration.
    override = write_config('override', {'useBestFitAsCentralValue': True})
    read_output(calculate('overrideResult', fitter_path, xsec, options=('-of', str(override))), 32, best_fit=True)
    calculate('dryRun', fitter_path, xsec, options=('--dry-run',))
    assert not Path('dryRun.root').exists()
    print('PASS CalcXsec exact propagation, normalization, central values and outputs')


if __name__ == '__main__':
    main()
