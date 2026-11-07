#!/usr/bin/env python3
"""Observable migration gradients against the complete smoothed likelihood."""
import copy
from importlib import import_module
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
fixture = import_module('590NumericsWeightGradient')


def check_threads(threads):
    config, build, check, _ = fixture.make_fixture(threads)
    li = config['likelihoodInterfaceConfig']
    li['jointProbabilityConfig']['ignoreBinsWithZeroPredictionAtPrior'] = False
    prop = li['propagatorConfig']
    sample = prop['sampleSetConfig']['sampleList'][0]
    definitions = prop['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions']
    # Same parameters affect both weights and observables. Composition is
    # (override or loaded X)*q + p, independently of dial declaration order.
    definitions[0]['dialSetDefinitions'].append({
        'dialType': 'Normalization', 'applyToVariable': 'X', 'variableResponseMode': 'Add',
        'dialResponseLimits': [-2., 2.], 'enableResponseSmoothing': True,
        'dialResponseSmoothingWidth': .1})
    definitions[1]['dialSetDefinitions'].append({
        'dialType': 'Normalization', 'applyToVariable': 'X', 'variableResponseMode': 'Multiply'})
    definitions[2]['dialSetDefinitions'].append({
        'dialType': 'Normalization', 'applyToVariable': 'X', 'variableResponseMode': 'Override',
        'applyCondition': 'X < .3'})
    for width in ({'width': .4}, {'widthFraction': .3}):
        sample['binningResponse'] = {'isEnabled': True, 'variables': [dict(name='X', **width)]}
        for normalized, eigen in ((False, False), (True, False), (False, True)):
            engine = build(config, normalized, eigen, analytic=True)
            data = engine.getLikelihoodInterface().getDataPropagator().getSampleSet().getSampleList()[0]
            before = [(b.sumWeights, b.sqrtSumSqWeights) for b in data.getHistogram().getBinContentList()]
            points = ([.1, -.13], [-.08, .05]) if eigen else (
                [.36, .92, 1.1], [-.1, .85, .45], [1.95, 1.05, 1.2])
            for point in points:
                check(engine, point)
            assert before == [(b.sumWeights, b.sqrtSumSqWeights) for b in data.getHistogram().getBinContentList()]
    # Clamped weight derivatives must not suppress observable migration.
    capped = copy.deepcopy(config)
    capped['likelihoodInterfaceConfig']['propagatorConfig']['globalEventReweightCap'] = .1
    engine = build(capped, False, False, analytic=True)
    check(engine, [.36, .92, 1.1])

    # Zero products are differentiated without division, for both weights
    # and observable multipliers. LeastSquares remains finite at zero yield.
    zero = copy.deepcopy(config)
    zero['likelihoodInterfaceConfig']['jointProbabilityConfig'] = {'type': 'LeastSquares'}
    zero_parameters = zero['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions']
    zero_parameters[1]['dialSetDefinitions'].append({
        'dialType': 'Normalization', 'applyToVariable': 'X', 'variableResponseMode': 'Multiply'})
    engine = build(zero, False, False, analytic=True)
    check(engine, [.36, 0., 1.1], one_sided=(1,))

    # Smoothing an unrelated coordinate does not make moving hard edges safe.
    bad = copy.deepcopy(config)
    bad_sample = bad['likelihoodInterfaceConfig']['propagatorConfig']['sampleSetConfig']['sampleList'][0]
    bad_sample['binning']['binningDefinition'].append({'name': 'rawQ3', 'edges': [-1., 2000.]})
    bad_parameters = bad['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions']
    bad_parameters[0]['dialSetDefinitions'].append({
        'dialType': 'Normalization', 'applyToVariable': 'rawQ3', 'variableResponseMode': 'Add'})
    try:
        build(bad, False, False, analytic=True)
    except RuntimeError as error:
        assert 'smoothing' in str(error).lower(), str(error)
    else:
        raise AssertionError('Moving unsmoothed axis was accepted')
    print(f'SUCCESS: smoothed observable gradients ({threads} threads)')


def main():
    for threads in (1, 4):
        check_threads(threads)


if __name__ == '__main__':
    main()
