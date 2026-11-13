#!/usr/bin/env python3
"""Weighted reaction breakdowns, plots and events must retain toy contributions.

Events in one bin have unequal weights AND different parameter responses.
Checking only bin totals would miss a uniform redistribution over those events.
"""
from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helpers'))
from CalcXsec import calculate, read_output, write_config, write_covariance
import numpy as np
import uproot

configurations = import_module('585NumericsCalcXsec').configurations


def main():
    events_path = Path('reactionEvents.root').resolve()
    inputs = {'x': np.array([1., 1., 4., 4.] * 2), 'y': np.full(8, 1.5),
              'category': np.repeat([0., 1.], 4), 'reaction': np.array([0, 1] * 4, dtype=np.int32),
              'weight': np.array([40., 80., 80., 160.] * 2)}
    with uproot.recreate(events_path) as output:
        output.mktree('events', {key: value.dtype for key, value in inputs.items()}).extend(inputs)
    fitter, xsec = configurations(events_path, write_covariance())
    definitions = fitter['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions']
    for reaction, definition in enumerate(definitions):
        definition['dialSetDefinitions'] = [
            {'dialType': 'Normalization', 'applyCondition': f'reaction == {reaction}'},
        ]
    likelihood = xsec['fitterEngineConfig']['likelihoodInterfaceConfig']
    likelihood['plotGeneratorConfig'] = {
        'isEnabled': True,
        'varDictionaries': [{'name': 'reaction', 'dictionary': [
            {'value': 0, 'title': 'CCQE', 'colorRoot': 801},
            {'value': 1, 'title': 'RES', 'colorRoot': 413},
        ]}],
        'histogramsDefinition': [
            {'varToPlot': 'Raw', 'useSampleBinning': True, 'rescaleAsBinWidth': False},
            {'varToPlot': 'x', 'useSampleBinning': True, 'rescaleAsBinWidth': True,
             'splitVarList': ['', 'reaction']},
        ],
    }
    likelihood['eventTreeWriter'] = {'isEnabled': True, 'writeDials': False}
    for sample in likelihood['propagatorConfig']['sampleSetConfig']['sampleList']:
        sample['normaliseParameterList'] = [{'name': 'targets', 'meanValue': 10., 'stdDev': 0.2}]
    fitter_path = write_config('fitter', fitter)
    for best_fit in (False, True):
        xsec['useBestFitAsCentralValue'] = best_fit
        path = calculate('outputs' + str(int(best_fit)), fitter_path, xsec)
        toys, nominal, _ = read_output(path, 32, best_fit=best_fit)
        central = nominal[:6] if best_fit else toys[:, :6].mean(axis=0)
        # Recover the shared normalization drawn in each toy, independently
        # from its two differently weighted reaction contributions.
        norm = (40 * toys[:, 6] + 80 * toys[:, 7]) / (6 * toys[:, 0])
        coefficients = np.full(2, 0.1) if best_fit else (toys[:, 6:] / norm[:, None]).mean(axis=0)
        with uproot.open(path) as output:
            for index, name in enumerate(('signalA (pre-fit)', 'signalB (pre-fit)')):
                expected = central[3*index:3*index+3] * [6, 12, 6]
                for role in ('model', 'data'):
                    tree = output[f'calcXsec/events/{role}/{name}/events_TTree']
                    events = tree['Event'].array(library='np')
                    leaves = tree['Leaves'].array(library='np')
                    # Check every event, not just totals: MC category fractions
                    # must retain the weighted contributions averaged over toys.
                    expected_weights = inputs['weight'][events['entryIndex']] * coefficients[leaves['reaction'].astype(int)]
                    np.testing.assert_allclose(events['eventWeight'], expected_weights, atol=1e-10)
                    for bin_index in range(3):
                        np.testing.assert_allclose(events['eventWeight'][events['sampleBinIndex'] == bin_index].sum(),
                                                   expected[bin_index], atol=1e-10)
                base = f'calcXsec/plots/histograms/{name}'
                for label in ('MC', 'Data'):
                    histogram = output[f'{base}/Raw/{label}_TH1D']
                    np.testing.assert_allclose(histogram.values(), expected, atol=1e-10)
                np.testing.assert_allclose(output[f'{base}/Raw/Data_TH1D'].errors(),
                                           toys[:, 3*index:3*index+3].std(axis=0) * [6, 12, 6], atol=1e-10)
                total = output[f'{base}/x/MC_TH1D'].values()
                expected_reactions = [np.array([40., 80., 0.]) * coefficients[0] / [2., 4., 2.],
                                      np.array([80., 160., 0.]) * coefficients[1] / [2., 4., 2.]]
                split_histograms = [obj for key, obj in output[f'{base}/x/reaction'].items()
                                    if getattr(obj, 'classname', '').startswith('TH1')]
                assert len(split_histograms) == 2
                for histogram, expected_reaction in zip(split_histograms, expected_reactions):
                    np.testing.assert_allclose(histogram.values(), expected_reaction, atol=1e-10)
                np.testing.assert_allclose(sum(hist.values() for hist in split_histograms), total, atol=1e-10)
            plot_histograms = [obj for _, obj in output['calcXsec/plots/histograms'].items(recursive=True)
                               if getattr(obj, 'classname', '').startswith('TH1')]
            for histogram in plot_histograms:
                assert np.isfinite(histogram.values()).all()
                assert np.isfinite(histogram.errors()).all()
            assert any(getattr(obj, 'classname', '') == 'TCanvas' for obj in output['calcXsec/plots/canvas'].values(recursive=True))
    print('PASS CalcXsec weighted reaction breakdowns, toy averages, events and plots')


if __name__ == '__main__':
    main()
