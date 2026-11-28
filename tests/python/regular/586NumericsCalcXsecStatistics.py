#!/usr/bin/env python3
"""All statistical switches, with independent analytic compound-Poisson moments."""
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
    fitter_path = write_config('fitter', fitter)
    prior = np.array([[0.01, -0.01], [-0.01, 0.04]])
    scale = np.array([60., 120.])  # volume * targets for each populated bin
    yields = np.array([120., 240.])
    sum_squared_weights = np.array([40.**2 + 80.**2, 80.**2 + 160.**2])
    no_stat_toys = None
    for stat, mc in ((False, False), (False, True), (True, False), (True, True)):
        config = deepcopy(xsec)
        config['xsecCalcConfig'] = {'enableStatThrowInToys': stat, 'enableEventMcThrow': mc}
        # Per-sample suppression must override the global MC switch.
        config['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig']['sampleSetConfig']['sampleList'][1]['disableEventMcThrow'] = True
        label = f'stat{int(stat)}mc{int(mc)}'
        toys = read_output(calculate(label, fitter_path, config, n=20000), 20000)[0]
        covariance = 4 * prior
        if stat:
            covariance = covariance + np.diag(yields / scale**2)
        if stat and mc:
            covariance = covariance + np.diag(sum_squared_weights * (1 + np.diag(prior)) / scale**2)
        check_moments(toys[:, :2], [2., 2.], covariance, label + ' sample A')
        suppressed_covariance = 4 * prior + (np.diag(yields / scale**2) if stat else 0)
        check_moments(toys[:, 3:5], [2., 2.], suppressed_covariance, label + ' sample B')
        np.testing.assert_array_equal(toys[:, [2, 5]], 0.)
        if not stat:
            if no_stat_toys is None:
                no_stat_toys = toys
            else:
                np.testing.assert_array_equal(no_stat_toys, toys)
        else:
            # Observed event counts are integer Poisson draws before division.
            np.testing.assert_allclose(toys[:, :2] * scale, np.rint(toys[:, :2] * scale), atol=1e-10)


if __name__ == '__main__':
    main()
