#!/usr/bin/env python3
"""Analytic Gaussian moments of linear and squared responses (20000 throws)."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fast'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helpers'))
from CalcXsec import calculate, check_moments, read_output, write_config, write_covariance, write_events
import numpy as np

configurations = import_module('585NumericsCalcXsec').configurations


def main():
    fitter, xsec = configurations(write_events(), write_covariance())
    prior = np.array([[0.01, -0.01], [-0.01, 0.04]])
    # Include xsec-xsec and xsec-parameter blocks across both samples.
    response = np.array([[2, 0], [0, 2], [0, 0], [2, 0], [0, 2], [0, 0], [1, 0], [0, 1]])
    toys, _, _ = read_output(calculate('linear', write_config('fitter', fitter), xsec, n=20000), 20000)
    check_moments(toys, response @ np.ones(2), response @ prior @ response.T, 'linear response')

    # Two multiplicative normalization dials give sigma_i = 2 p_i^2. Their exact Gaussian covariance is
    # Cov(p_i^2,p_j^2) = 4 mu_i mu_j C_ij + 2 C_ij^2.
    squared = deepcopy(fitter)
    definitions = squared['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions']
    for definition in definitions:
        definition['dialSetDefinitions'] *= 2
    squared_path = write_config('squaredFitter', squared)
    toys, nominal, _ = read_output(calculate('squared', squared_path, xsec, n=20000), 20000)
    np.testing.assert_allclose(toys[:, :2], 2 * toys[:, 6:] ** 2)
    square_mean = 1 + np.diag(prior)
    square_covariance = 4 * prior + 2 * prior ** 2
    check_moments(toys[:, :2], 2 * square_mean, 4 * square_covariance, 'squared response')
    np.testing.assert_allclose(nominal[:2], [2., 2.])
    assert np.all(toys[:, :2].mean(axis=0) > nominal[:2])

    # Best-fit central values leave the throw covariance unchanged, even for
    # a nonlinear response where sigma(E[p]) differs from E[sigma(p)].
    central = deepcopy(xsec)
    central['useBestFitAsCentralValue'] = True
    bf_toys = read_output(calculate('squaredBestFit', squared_path, central, n=20000), 20000, best_fit=True)[0]
    np.testing.assert_array_equal(bf_toys, toys)

    # Recover the denominator from each recorded throw to test the Gaussian
    # normalizer itself, without assuming that its inverse is Gaussian.
    for shared in (True, False):
        normalized = deepcopy(xsec)
        samples = normalized['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig']['sampleSetConfig']['sampleList']
        for index, sample in enumerate(samples):
            sample['normaliseParameterList'] = [
                {'name': 'targets' if shared else f'targets{index}', 'meanValue': 10., 'stdDev': 0.2},
            ]
        values = read_output(calculate('normalizer' + str(shared), write_config('fitter', fitter), normalized, n=20000), 20000)[0]
        denominators = 20 * values[:, [6, 6]] / values[:, [0, 3]]
        expected_covariance = np.full((2, 2), 0.04) if shared else np.eye(2) * 0.04
        check_moments(denominators, [10., 10.], expected_covariance, f'normalizers shared={shared}')


if __name__ == '__main__':
    main()
