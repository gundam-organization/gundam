#!/usr/bin/env python3
"""External Gaussian formula gradients through physical, normalized and eigen fits."""
import copy
from importlib import import_module
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
fixture = import_module('590NumericsWeightGradient')


def check_threads(threads):
    import GUNDAM
    config, build, check, close = fixture.make_fixture(threads)
    measurement = {
        'name': 'nonlinearMeasurement',
        # Alias order intentionally differs from ROOT's formula parameter order.
        'parameters': [
            {'name': 'second', 'parameterSetName': 'parameters', 'parameterIndex': 1},
            {'name': 'again', 'parameterSetName': 'parameters', 'parameterName': 'p'},
            {'name': 'first', 'parameterSetName': 'parameters', 'parameterIndex': 0}],
        'formula': 'sin([first])+[again]*[second]*[second]',
        'type': {'name': 'gaussian', 'mean': .8, 'stdDev': .3}}
    other = copy.deepcopy(measurement)
    other.update(name='otherMeasurement', formula='[first]+[again]-[second]')
    other['type'] = {'name': 'gaussian', 'mean': -.2, 'stdDev': .7}
    external = {'isEnabled': True, 'constraintsList': [
        measurement, other, {'name': 'disabled', 'isEnabled': False}]}
    config['likelihoodInterfaceConfig']['externalConstraints'] = external
    for normalized in (False, True):
        for eigen in (False, True):
            engine = build(config, normalized, eigen, analytic=True)
            points = fixture.nearby_fit_points(engine, normalized) if eigen else ([.4, .9, 1.1], [.2, 1.05, .95])
            for point in points:
                check(engine, point)

    # Flat parameters have no internal Gaussian prior but retain the measurement.
    flat = copy.deepcopy(config)
    definitions = flat['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions']
    definitions[0]['priorType'] = 'Flat'
    engine = build(flat, False, False, analytic=True)
    check(engine, [.4, .9, 1.1])

    # Arbitrary physical-coordinate requests also support fixed parameters and
    # subsets. Compare total gradients to full-likelihood finite differences.
    likelihood = engine.getLikelihoodInterface()
    parameters = likelihood.getModelPropagator().getParametersManager().getParameterSetsList()[0].getParameterList()
    parameter = parameters[0]
    parameter.setIsFixed(True)
    subset = GUNDAM.ParameterGradient()
    subset.resetValue(parameter)
    likelihood.evalGradient(subset)
    analytic = subset.getValue(parameter)
    point = parameter.getParameterValue()
    parameter.setParameterValue(point + 1e-5)
    likelihood.propagateAndEvalLikelihood()
    upper = likelihood.getLastLikelihood()
    parameter.setParameterValue(point - 1e-5)
    likelihood.propagateAndEvalLikelihood()
    lower = likelihood.getLastLikelihood()
    parameter.setParameterValue(point)
    close(analytic, (upper - lower) / 2e-5)

    # Gradient validation is strict at initialization; value-only fits keep
    # accepting formulas for which ROOT/Clad would use numerical differences.
    unsupported = copy.deepcopy(config)
    unsupported['likelihoodInterfaceConfig']['externalConstraints']['constraintsList'][0]['formula'] = (
        'TMath::BesselJ0([first])+[again]+[second]')
    build(unsupported, False, False)
    try:
        build(unsupported, False, False, analytic=True)
    except RuntimeError as error:
        assert 'nonlinearMeasurement' in str(error), str(error)
        assert 'analytic gradient unavailable' in str(error), str(error)
    else:
        raise AssertionError('Unsupported external formula gradient was accepted')
    unsupported['likelihoodInterfaceConfig']['externalConstraints']['isEnabled'] = False
    engine = build(unsupported, False, False, analytic=True)
    check(engine, [.4, .9, 1.1])
    print(f'SUCCESS: external constraint gradients ({threads} threads)')


def main():
    for threads in (1, 4):
        check_threads(threads)


if __name__ == '__main__':
    main()
