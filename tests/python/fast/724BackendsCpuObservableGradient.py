#!/usr/bin/env python3
"""CPU observable/fraction gradients against native execution and finite differences."""
import copy
from importlib import import_module
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
fixture = import_module('590NumericsWeightGradient')
observables = import_module('595NumericsObservableGradient')


def check_threads(threads):
    config, build, check, close = fixture.make_fixture(threads)
    li = config['likelihoodInterfaceConfig']
    observables.add_reference_events(li['dataSetList'][0]['model']['filePathList'][0])
    li['jointProbabilityConfig']['ignoreBinsWithZeroPredictionAtPrior'] = False
    prop = li['propagatorConfig']
    sample = prop['sampleSetConfig']['sampleList'][0]
    sample['useDynamicBinning'] = True
    definitions = prop['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions']
    for definition in definitions[1:]:
        definition['dialSetDefinitions'][0]['applyCondition'] = 'rawQ3 >= 0'
    for index, mode in enumerate(('Add', 'Multiply', 'Override')):
        dial = {'dialType': 'Normalization', 'applyToVariable': 'X',
                'variableResponseMode': mode, 'applyCondition': 'rawQ3 >= 0'}
        if mode == 'Override':
            dial['applyCondition'] += ' && X < .3'
        if mode == 'Add':
            dial.update(dialResponseLimits=[-2., 2.], enableResponseSmoothing=True,
                        dialResponseSmoothingWidth=.1)
        definitions[index]['dialSetDefinitions'].append(dial)
    # Repeated multipliers exercise product differentiation, including zeros.
    definitions[1]['dialSetDefinitions'].append(copy.deepcopy(definitions[1]['dialSetDefinitions'][-1]))
    for width in ({'width': .4}, {'widthFraction': .3}):
        sample['binningResponse'] = {'isEnabled': True, 'variables': [dict(name='X', **width)]}
        for normalized, eigen in ((False, False), (True, False), (False, True), (True, True)):
            native = build(config, normalized, eigen, analytic=True)
            cpu_config = copy.deepcopy(config)
            cpu_config['likelihoodInterfaceConfig']['backendManagerConfig'] = {'isEnabled': True, 'useBackend': 'CPU'}
            cpu = build(cpu_config, normalized, eigen, analytic=True)
            cpu.getBackendsManager().setEnableAutoMaterialize(False)
            points = fixture.nearby_fit_points(cpu, normalized) if normalized or eigen else (
                [.36, .92, 1.1], [-.1, .85, .45], [1.95, 1.05, 1.2], [.36, .92, 1.1])
            for point in points:
                expected = check(native, point)
                actual = check(cpu, point, preserve_likelihood=False)
                for a, b in zip(actual, expected):
                    close(a, b)
            # Materializing/deactivating retains the same model and derivatives.
            cpu.getBackendsManager().deactivateBackend()
            for a, b in zip(check(cpu, points[0]), check(native, points[0])):
                close(a, b)
            cpu.getBackendsManager().activate('CPU')
            check(cpu, points[0], preserve_likelihood=False)

    for cap in (False, True):
        cfg = copy.deepcopy(config)
        cfg['likelihoodInterfaceConfig']['backendManagerConfig'] = {'isEnabled': True, 'useBackend': 'CPU'}
        if cap:
            cfg['likelihoodInterfaceConfig']['propagatorConfig']['globalEventReweightCap'] = .1
        cpu = build(cfg, False, False, analytic=True)
        check(cpu, [.36, .92, 1.1], preserve_likelihood=False)
        check(cpu, [.36, 0., 1.1], one_sided=(1,), preserve_likelihood=False)

    # Every moving histogram axis must be smoothed, even if another axis is.
    for missing in ('smoothing', 'moving-axis'):
        bad = copy.deepcopy(config)
        bad_li = bad['likelihoodInterfaceConfig']
        bad_li['backendManagerConfig'] = {'isEnabled': True, 'useBackend': 'CPU'}
        bad_prop = bad_li['propagatorConfig']
        bad_sample = bad_prop['sampleSetConfig']['sampleList'][0]
        if missing == 'smoothing':
            bad_sample.pop('binningResponse')
        else:
            bad_sample['binning']['binningDefinition'].append({'name': 'rawQ3', 'edges': [-1., 2000.]})
            bad_prop['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions'][0]['dialSetDefinitions'].append({
                'dialType': 'Normalization', 'applyToVariable': 'rawQ3', 'variableResponseMode': 'Add'})
        try:
            build(bad, False, False, analytic=True)
        except RuntimeError as error:
            assert 'smoothing' in str(error).lower(), str(error)
        else:
            raise AssertionError(f'Accepted observable gradient without {missing}')
    print(f'SUCCESS: CPU observable and fractional gradients ({threads} threads)')


if __name__ == '__main__':
    for threads in (1, 4):
        check_threads(threads)
