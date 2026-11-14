#!/usr/bin/env python3
"""Real fitter -> CalcXsec chain, prior vs posterior and Gaussian covariance injection."""
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helpers'))
from CalcXsec import calculate, executable, read_output, run, write_config, write_covariance, write_events
import numpy as np
import uproot

configurations = import_module('585NumericsCalcXsec').configurations


def main():
    fitter, xsec = configurations(write_events(), write_covariance())
    fitter.update(fit=True, scanParameters=False, generateOneSigmaPlots=False)
    engine = fitter['fitterEngineConfig']
    engine['minimizerConfig'] = {'minimizer': 'Minuit2', 'algorithm': 'Migrad',
                               'errors': 'Hesse', 'tolerance': 1.e-7, 'print_level': 0}
    engine['generateSamplePlots'] = False
    likelihood = engine['likelihoodInterfaceConfig']
    dataset = likelihood['datasetList'][0]
    dataset['selectedDataEntry'] = 'observed'
    dataset['data'] = [dict(dataset['model'], name='observed', nominalWeightFormula='1.2*weight')]
    likelihood['propagatorConfig']['sampleSetConfig']['sampleList'][0]['binning'] = {
        'binningDefinition': [{'name': 'x', 'edges': [0, 2, 6]}],
    }
    fitter_path = write_config('fitter', fitter)
    fit_output = Path('fit.root').resolve()
    run([executable('gundamFitter'), '-c', str(fitter_path), '-o', str(fit_output),
         '-s', '12345', '-t', '1', '--light-mode'], 'fit')
    with uproot.open(fit_output) as output:
        state = json.loads(output['FitterEngine/postFit/parState_TNamed'].member('fTitle'))
        print('Posterior parameter state:', state)
        definitions = state['parameterSetList'][0]['parameterValues']
        best_fit = np.array([definition['value'] for definition in definitions])
        posterior = output['FitterEngine/postFit/Hesse/hessian/postfitCovarianceOriginal_TH2D'].values()
    prior = np.array([[0.01, -0.01], [-0.01, 0.04]])
    yields = np.array([240., 480.])
    observed = 1.2 * yields
    # Independently solve the stationarity equations of Poisson + Gaussian prior.
    reference = np.ones(2)
    precision = np.linalg.inv(prior)
    for _ in range(15):
        gradient = yields - observed / reference + precision @ (reference - 1)
        hessian = np.diag(observed / reference**2) + precision
        reference -= np.linalg.solve(hessian, gradient)
    np.testing.assert_allclose(best_fit, reference, rtol=2e-5)
    np.testing.assert_allclose(posterior, np.linalg.inv(np.diag(observed / reference**2) + precision), rtol=2e-3)
    assert np.max(np.abs(best_fit - 1)) > 0.05

    prefit_toys, _, _ = read_output(calculate('prefitConfig', fitter_path, xsec), 32)
    root_prefit = read_output(calculate('prefitRoot', fit_output, xsec, options=('--use-prefit',)), 32)[0]
    np.testing.assert_allclose(root_prefit, prefit_toys, atol=1e-10)
    postfit_toys, nominal, _ = read_output(calculate('postfit', fit_output, xsec), 32)
    np.testing.assert_allclose(nominal[6:], best_fit, atol=1e-10)
    np.testing.assert_allclose(nominal[:6], best_fit[[0, 1, 1, 0, 1, 1]] * [2, 2, 0, 2, 2, 0])
    np.testing.assert_allclose(postfit_toys[:, :2], 2 * postfit_toys[:, 6:])
    # Same seed gives the same standard-normal stream: compare the actual
    # transforms, which tests covariance injection with only 32 throws.
    z = np.linalg.solve(np.linalg.cholesky(prior), (prefit_toys[:, 6:] - 1).T)
    np.testing.assert_allclose(postfit_toys[:, 6:], best_fit + (np.linalg.cholesky(posterior) @ z).T, atol=1e-9)
    central = deepcopy(xsec)
    central['useBestFitAsCentralValue'] = True
    read_output(calculate('postfitCentral', fit_output, central), 32, best_fit=True)
    print('PASS fitter -> CalcXsec, posterior values/covariance, explicit prefit')


if __name__ == '__main__':
    main()
