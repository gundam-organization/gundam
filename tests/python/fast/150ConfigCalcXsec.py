#!/usr/bin/env python3
"""CalcXsec invalid inputs must fail with an actionable diagnostic."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helpers'))
from CalcXsec import calculate, executable, run, write_config, write_covariance, write_events

configurations = import_module('585NumericsCalcXsec').configurations


def main():
    fitter, xsec = configurations(write_events(), write_covariance())
    fitter_path = write_config('fitter', fitter)
    xsec_path = write_config('xsec', xsec)
    app = executable('gundamCalcXsec')
    for label, arguments, diagnostic in (
        ('missingConfig', ['-f', str(fitter_path), '-n', '2'], 'Xsec calculator config file not provided'),
        ('missingFitter', ['-c', str(xsec_path), '-n', '2'], 'Did not provide the output fitter file'),
        ('missingToys', ['-c', str(xsec_path), '-f', str(fitter_path)], 'Did not provide number of toys'),
    ):
        run([app, *arguments], label, error=diagnostic)
    for count in (0, -1):
        calculate('invalidToys' + str(count), fitter_path, xsec, n=count, error='Number of toys must be positive')
    Path('invalid.root').write_text('Not a ROOT file\n')
    calculate('invalidRoot', Path('invalid.root').resolve(), xsec, error='Could not open fitter output file')
    for label, normalizer, message in (
        ('zeroMean', {'name': 'bad', 'meanValue': 0}, 'Normalizer mean must be positive'),
        ('negativeSigma', {'name': 'bad', 'meanValue': 10, 'stdDev': -1}, 'Normalizer stdDev must be non-negative'),
    ):
        config = deepcopy(xsec)
        sample = config['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig']['sampleSetConfig']['sampleList'][0]
        sample['normaliseParameterList'] = [normalizer]
        calculate(label, fitter_path, config, error=message)
    config = deepcopy(xsec)
    sample = config['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig']['sampleSetConfig']['sampleList'][0]
    del sample['binning']
    sample['parSetBinning'] = 'unknown'
    calculate('unknownParameterSet', fitter_path, config, error='Could not find unknown among fit dial collections')
    print('PASS CalcXsec invalid inputs and diagnostics')


if __name__ == '__main__':
    main()
