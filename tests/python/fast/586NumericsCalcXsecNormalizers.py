#!/usr/bin/env python3
"""Flux integrals, parameter binning, condition variables and shared normalizers."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helpers'))
from CalcXsec import calculate, read_output, write_config, write_covariance, write_events
import numpy as np
import uproot

configurations = import_module('585NumericsCalcXsec').configurations


def main():
    fitter, xsec = configurations(write_events(), write_covariance())
    fitter_path = write_config('fitter', fitter)
    samples = xsec['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig']['sampleSetConfig']['sampleList']

    # One standard normal deviate per name and toy, shared across samples.
    for sample in samples:
        sample['normaliseParameterList'] = [{'name': 'targets', 'meanValue': 10., 'stdDev': 0.2}]
    toys, nominal, _ = read_output(calculate('shared', fitter_path, xsec), 32)
    effective_norm = 20 * toys[:, [6, 7, 6, 7]] / toys[:, [0, 1, 3, 4]]
    np.testing.assert_allclose(effective_norm, np.repeat(effective_norm[:, :1], 4, axis=1), atol=1e-10)
    assert effective_norm[:, 0].std() > 0.05
    np.testing.assert_allclose(nominal[:6], [2, 2, 0, 2, 2, 0])
    samples[1]['normaliseParameterList'][0]['name'] = 'otherTargets'
    independent = read_output(calculate('independent', fitter_path, xsec), 32)[0]
    assert not np.allclose(independent[:, 0], independent[:, 3])

    # A whole-set normalization dial associates each parameter with a flux bin.
    # The condition category is NOT a volume dimension. An unmatched flux bin
    # at x=9 must not enter the integral, even with an enormous content.
    binning = Path('fluxBinning.txt').resolve()
    binning.write_text('variables: category x x\n0 0 2\n0 2 6\n')
    flux_path = Path('fluxInput.root').resolve()
    with uproot.recreate(flux_path) as output:
        output['flux'] = (np.array([10., 30., 1.e9]), np.array([0., 2., 6., 12.]))
    parset = fitter['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]
    for definition in parset['parameterDefinitions']:
        del definition['dialSetDefinitions']
    parset['dialSetDefinitions'] = [{'dialType': 'Normalization', 'parametersBinningPath': str(binning)}]
    parset['normalisations'] = [{
        'name': 'fluxIntegral', 'filePath': str(flux_path), 'histogramPath': 'flux',
        'axisVariable': 'x', 'parSelections': [{'name': 'category', 'value': 0}],
    }]
    flux_config = {
        'xsecCalcConfig': {'enableStatThrowInToys': False, 'enableEventMcThrow': False},
        'fitterEngineConfig': {'likelihoodInterfaceConfig': {'propagatorConfig': {
            'sampleSetConfig': {'sampleList': [{
                'name': 'fluxSignal', 'selectionCutStr': 'category == 0',
                'parSetBinning': 'response',
                'normaliseParameterList': [{'name': 'flux', 'parSetNormName': 'fluxIntegral'}],
            }]},
        }}},
    }
    fitter_path = write_config('fluxFitter', fitter)
    flux_toys, nominal, _ = read_output(calculate('flux', fitter_path, flux_config), 32)
    norm = 10 * flux_toys[:, 2] + 30 * flux_toys[:, 3]
    np.testing.assert_allclose(flux_toys[:, 0], 60 * flux_toys[:, 2] / norm)
    np.testing.assert_allclose(flux_toys[:, 1], 60 * flux_toys[:, 3] / norm)
    np.testing.assert_allclose(nominal[:2], [1.5, 1.5])

    # Explicit binning and binning inherited from the parameter set agree.
    explicit = deepcopy(flux_config)
    sample = explicit['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig']['sampleSetConfig']['sampleList'][0]
    del sample['parSetBinning']
    sample['binning'] = str(binning)
    np.testing.assert_array_equal(read_output(calculate('explicit', fitter_path, explicit), 32)[0], flux_toys)

    # Inline parameter binning has no filename: inheritance must keep its config.
    inline_fitter = deepcopy(fitter)
    inline_set = inline_fitter['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]
    inline_set['dialSetDefinitions'][0]['parametersBinningPath'] = {
        'binningDefinition': [{'name': 'x', 'edges': [0, 2, 6]}],
    }
    np.testing.assert_array_equal(
        read_output(calculate('inline', write_config('inlineFitter', inline_fitter), flux_config), 32)[0], flux_toys)

    # Selecting a single flux bin cancels its parameter exactly in sigma_0.
    single_flux_path = Path('singleFlux.root').resolve()
    with uproot.recreate(single_flux_path) as output:
        output['flux'] = (np.array([10., 0.]), np.array([0., 2., 6.]))
    parset['normalisations'][0]['filePath'] = str(single_flux_path)
    cancel = read_output(calculate('cancel', write_config('cancelFitter', fitter), flux_config), 32)[0]
    np.testing.assert_allclose(cancel[:, 0], 6., atol=1e-12)

    bad = deepcopy(flux_config)
    bad['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig']['sampleSetConfig']['sampleList'][0]['normaliseParameterList'][0]['parSetNormName'] = 'missing'
    calculate('missingNormalizer', fitter_path, bad, error='Could not find parSetNorm obj')
    invalid_flux = deepcopy(fitter)
    normalizer = invalid_flux['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['normalisations'][0]
    # No parameter bin has category=1, so this is an empty flux selection.
    normalizer['parSelections'] = [{'name': 'category', 'value': 1}]
    calculate('zeroFlux', write_config('zeroFluxFitter', invalid_flux), flux_config, error='Invalid flux normalization')
    normalizer['filePath'] = str(Path('missingFlux.root').resolve())
    calculate('missingFluxFile', write_config('missingFluxFitter', invalid_flux), flux_config,
              error='Could not open normalization file')
    print('PASS CalcXsec fluctuating normalizers, flux integrals and parameter binning')


if __name__ == '__main__':
    main()
