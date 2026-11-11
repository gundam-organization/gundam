#!/usr/bin/env python3
"""Compare resident backend weight gradients with the original engine and finite differences."""
import argparse
import copy
from importlib import import_module
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
checks = import_module('590NumericsWeightGradient')


def check_threads(threads, backend_name):
    config, build, check, close = checks.make_fixture(threads)
    splines = copy.deepcopy(config)
    splines['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions'][0]['dialSetDefinitions'][0]['dialType'] = 'Spline'
    mixed = copy.deepcopy(config)
    mixed['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions'][1]['priorType'] = 'Flat'
    for normalized in (False, True):
        for eigen in (False, True):
            for configuration in (config, splines, mixed):
                if configuration is mixed and not eigen:
                    continue
                engine = build(configuration, normalized, eigen)
                minimizer = engine.getMinimizer()
                parameters = list(minimizer.getMinimizerFitParameterList())
                point = [.07*(i+1) if normalized else p.getParameterValue()+.07*p.getStdDevValue()
                         for i, p in enumerate(parameters)]
                reference = check(engine, point)
                manager = engine.getBackendsManager()
                manager.activate(backend_name)
                manager.setEnableAutoMaterialize(False)
                actual = check(engine, point, preserve_likelihood=False)
                for a, b in zip(actual, reference):
                    close(a, b)
                manager.deactivateBackend()
                for a, b in zip(check(engine, point), reference):
                    close(a, b)

    if backend_name == 'CPU':
        # Cached structural validation must be invalidated by a data update.
        engine = build(config, False, False)
        manager = engine.getBackendsManager()
        manager.activate('CPU')
        check(engine, [.4, 1., 1.], preserve_likelihood=False)
        sample = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()[0]
        sample.setUseDynamicBinning(True)
        manager.updateDataHistograms()
        try:
            engine.getMinimizer().evalFitGradientAt([.4, 1., 1.])
        except RuntimeError as error:
            assert 'smoothing' in str(error).lower(), str(error)
        else:
            raise AssertionError('Data update retained stale gradient validation')
        sample.setUseDynamicBinning(False)
        manager.updateDataHistograms()
        check(engine, [.4, 1., 1.], preserve_likelihood=False)

    least_squares = copy.deepcopy(config)
    least_squares['likelihoodInterfaceConfig']['jointProbabilityConfig'] = {'type': 'LeastSquares'}
    engine = build(least_squares, False, False)
    engine.getBackendsManager().activate(backend_name)
    try:
        engine.getMinimizer().evalFitGradientAt([.4, 1., 1.])
    except RuntimeError as error:
        assert 'BarlowBeestonLiteGaussian' in str(error), str(error)
    else:
        raise AssertionError('Unsupported backend likelihood gradient was accepted')

    # CPU and MPS differentiate compiled formulas without ROOT/Clad.
    formula = copy.deepcopy(config)
    formula['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions'][0]['dialSetDefinitions'] = [
        checks.helpers.formula('formula', '1+.1*[arg0]', {'parameterName': 'p'})]
    rejected = [(least_squares, 'BarlowBeestonLiteGaussian')]
    if backend_name in ('CPU', 'MPS'):
        for normalized, eigen in ((False, False), (True, False), (False, True), (True, True)):
            cfg = copy.deepcopy(formula)
            cfg['likelihoodInterfaceConfig']['backendManagerConfig'] = {'isEnabled': True, 'useBackend': backend_name}
            engine = build(cfg, normalized, eigen, analytic=True)
            parameters = list(engine.getMinimizer().getMinimizerFitParameterList())
            point = [.07*(i+1) if normalized else p.getParameterValue()+.07*p.getStdDevValue()
                     for i, p in enumerate(parameters)]
            check(engine, point, preserve_likelihood=False)
    else:
        rejected.append((formula, 'Unsupported'))
    for configuration, message in rejected:
        cfg = copy.deepcopy(configuration)
        cfg['likelihoodInterfaceConfig']['backendManagerConfig'] = {'isEnabled': True, 'useBackend': backend_name}
        try:
            build(cfg, False, False, analytic=True)
        except RuntimeError as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError('Incompatible analytic backend configuration initialized')
    print(f'SUCCESS: backend weight gradients ({threads} threads, {backend_name})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('testDirectory', nargs='?', help=argparse.SUPPRESS)
    parser.add_argument('--backend', choices=['CPU', 'MPS', 'CUDA'], default='CPU')
    args = parser.parse_args()
    for threads in (1, 4):
        check_threads(threads, args.backend)


if __name__ == '__main__':
    main()
