#!/usr/bin/env python3
"""CPU Gaussian Barlow-Beeston gradients in serial and parallel propagation."""
import copy
from importlib import import_module
import json
import math
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)
helpers = import_module('230CompositeDials')


def check_threads(threads):
    import GUNDAM
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(threads)
    path = Path.cwd() / '903CpuWeightGradient.root'
    helpers.write_input(path)
    macro = path.with_suffix('.C')
    macro.write_text('''#include <TFile.h>
#include <TMatrixDSym.h>
#include <TSystem.h>
{ {
 TFile file(INPUT, "UPDATE");
 TMatrixDSym covariance(3);
 double sigma[3] = {.5, .3, .7};
 double corr[3][3] = {{1., .25, -.15}, {.25, 1., .35}, {-.15, .35, 1.}};
 for(int i=0;i<3;++i) for(int j=0;j<3;++j) covariance(i,j)=sigma[i]*sigma[j]*corr[i][j];
 covariance.Write("covariance");
 } gSystem->Exit(0); }
'''.replace('INPUT', json.dumps(str(path))))
    subprocess.run(['root', '-n', '-l', '-b', '-q', str(macro)], check=True)
    config = helpers.make_config(path, {})
    li = config['likelihoodInterfaceConfig']
    li['jointProbabilityConfig'] = {'type': 'BarlowBeestonLiteGaussian', 'ignoreBinsWithZeroPredictionAtPrior': True}
    # Fourth bin is empty at the prior and must remain ignored in the gradient.
    li['propagatorConfig']['sampleSetConfig']['sampleList'][0]['binning']['binningDefinition'][0]['edges'].append(4)
    parameters = li['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]
    parameters.pop('dialSetDefinitions')
    parameters.update(parameterDefinitionFilePath=str(path), covarianceMatrix='covariance',
                      parameterDefinitions=[
        {'name': 'p', 'parameterIndex': 0, 'priorValue': .3, 'dialSetDefinitions': [{'dialType': 'Graph', 'dialLeafName': 'graph'}]},
        {'name': 'q', 'parameterIndex': 1, 'priorValue': 1., 'dialSetDefinitions': [{'dialType': 'Normalization'}]},
        {'name': 'r', 'parameterIndex': 2, 'priorValue': 1., 'dialSetDefinitions': [{'dialType': 'Normalization', 'applyCondition': 'X < 1.5'}]}])

    def build(configuration, normalized, eigen, backend=None, analytic=False):
        cfg = copy.deepcopy(configuration)
        cfg['minimizerConfig'] = {'useNormalizedFitSpace': normalized, 'writeLlhHistory': False,
                                  'useAnalyticGradient': analytic}
        if backend is not None:
            cfg['likelihoodInterfaceConfig']['backendManagerConfig'] = {'isEnabled': True, 'useBackend': backend}
        cfg['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['enableEigenDecomp'] = eigen
        engine = GUNDAM.FitterEngine()
        engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(cfg))))
        engine.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
        engine.initialize()
        return engine

    def close(actual, expected):
        assert math.isfinite(actual) and math.isfinite(expected), (actual, expected)
        assert math.isclose(actual, expected, rel_tol=3e-5, abs_tol=3e-6), (actual, expected)

    def check(engine, point, one_sided=()):
        minimizer = engine.getMinimizer()
        # Deliberately leave the engine at another point: the callback must refresh it.
        minimizer.evalFit([value + .02 for value in point])
        gradient = minimizer.evalFitGradientAt(point)
        likelihood = engine.getLikelihoodInterface()
        total, stat, penalty = likelihood.getLastLikelihood(), likelihood.getLastStatLikelihood(), likelihood.getLastPenaltyLikelihood()
        close(total, minimizer.evalFit(point))
        close(stat, likelihood.getLastStatLikelihood())
        close(penalty, likelihood.getLastPenaltyLikelihood())
        assert len(gradient) == len(point)
        for i, value in enumerate(gradient):
            step = 2e-5 * max(1., abs(point[i]))
            up, down = list(point), list(point)
            up[i] += step
            down[i] -= step
            if i in one_sided:
                # Normalization response is clamped at zero: use its right derivative.
                twice = list(point)
                twice[i] += 2*step
                numeric = (-3*minimizer.evalFit(point) + 4*minimizer.evalFit(up) - minimizer.evalFit(twice))/(2*step)
            else:
                numeric = (minimizer.evalFit(up) - minimizer.evalFit(down))/(2*step)
            close(value, numeric)
        # The legacy interface also refreshes current parameters without destroying the LLH.
        minimizer.evalFit(point)
        before = likelihood.getLastLikelihood()
        minimizer.evalFitGradient()
        close(likelihood.getLastLikelihood(), before)
        close(likelihood.getLastStatLikelihood(), stat)
        return gradient

    for normalized in (False, True):
        for eigen in (False, True):
            engine = build(config, normalized, eigen)
            pars = list(engine.getMinimizer().getMinimizerFitParameterList())
            assert len(pars) == 3
            point = [0. if normalized else p.getParameterValue() for p in pars]
            check(engine, point)
            shifted = [value + sign*.13*(1. if normalized else p.getStdDevValue())
                       for value, sign, p in zip(point, (1., -1., 1.), pars)]
            check(engine, shifted)
            if not eigen:
                # Away from knots, clamped Graph input has exactly zero statistical slope.
                clamped = list(shifted)
                clamped[0] = (3.-pars[0].getPriorValue())/pars[0].getStdDevValue() if normalized else 3.
                check(engine, clamped)
            try:
                engine.getMinimizer().evalFitGradientAt(point[:-1])
            except ValueError:
                pass
            else:
                raise AssertionError('Wrong gradient dimension accepted')

    # Eigenvectors span only constrained parameters, not the remaining free coordinates.
    mixed = copy.deepcopy(config)
    mixed['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions'][1]['priorType'] = 'Flat'
    for normalized in (False, True):
        engine = build(mixed, normalized, True)
        assert len(engine.getMinimizer().getMinimizerFitParameterList()) == 2
        original = engine.getLikelihoodInterface().getModelPropagator().getParametersManager().getParameterSetsList()[0].getParameterList()
        free_parameter = original[1]
        close(free_parameter.getParameterValue(), 1.)
        # A free coordinate must not enter the two-dimensional eigen transform,
        # even when its value differs from its prior.
        free_parameter.setParameterValue(.87)
        check(engine, [.1, -.13])
        close(free_parameter.getParameterValue(), .87)
    # Exercise spline dials through the same Gaussian BB histogram chain.
    splines = copy.deepcopy(config)
    splines['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions'][0]['dialSetDefinitions'][0]['dialType'] = 'Spline'
    engine = build(splines, False, False)
    check(engine, [.43, .95, 1.08])

    # Linear product derivatives must handle one and multiple zero dial responses.
    least_squares = copy.deepcopy(config)
    least_squares['likelihoodInterfaceConfig']['jointProbabilityConfig'] = {'type': 'LeastSquares'}
    engine = build(least_squares, False, False)
    check(engine, [.3, 0., 1.], (1,))
    check(engine, [.3, 0., 0.], (1, 2))

    unsupported = copy.deepcopy(config)
    unsupported['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions'][0]['dialSetDefinitions'] = [helpers.formula('unsupported', '1+.1*[arg0]', {'parameterName': 'p'})]
    engine = build(unsupported, False, False)
    try:
        engine.getMinimizer().evalFitGradientAt([.4, 1., 1.])
    except RuntimeError as error:
        assert 'unsupported' in str(error)
    else:
        raise AssertionError('Unsupported dial silently differentiated')
    for kind in ('smooth', 'observable'):
        cfg = copy.deepcopy(config)
        prop = cfg['likelihoodInterfaceConfig']['propagatorConfig']
        if kind == 'smooth':
            prop['sampleSetConfig']['sampleList'][0]['binningResponse'] = {
                'isEnabled': True, 'variables': [{'name': 'X', 'width': .2}]}
        else:
            prop['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions'][0]['dialSetDefinitions'][0].update(
                applyToVariable='X', variableResponseMode='Override')
        engine = build(cfg, False, False)
        try:
            engine.getMinimizer().evalFitGradientAt([.4, 1., 1.])
        except RuntimeError as error:
            assert 'bins' in str(error) or 'observable' in str(error), str(error)
        else:
            raise AssertionError('Unsupported binning/observable gradient was accepted')
    if hasattr(engine, 'getBackendsManager'):
        for normalized in (False, True):
            for eigen in (False, True):
                for configuration in (config, splines, mixed):
                    if configuration is mixed and not eigen:
                        continue
                    engine = build(configuration, normalized, eigen)
                    minimizer = engine.getMinimizer()
                    pars = list(minimizer.getMinimizerFitParameterList())
                    point = [.07*(i+1) if normalized else p.getParameterValue()+.07*p.getStdDevValue()
                             for i, p in enumerate(pars)]
                    reference = check(engine, point)
                    manager = engine.getBackendsManager()
                    manager.activate('CPU')
                    manager.setEnableAutoMaterialize(False)
                    actual = check(engine, point)
                    for a, b in zip(actual, reference): close(a, b)
                    # Switching engines must not reuse a stale gradient result.
                    manager.deactivateBackend()
                    for a, b in zip(check(engine, point), reference): close(a, b)
        # Unsupported CPU likelihoods fail explicitly, without host fallback.
        engine = build(least_squares, False, False)
        engine.getBackendsManager().activate('CPU')
        try:
            engine.getMinimizer().evalFitGradientAt([.4, 1., 1.])
        except RuntimeError as error:
            assert 'BarlowBeestonLiteGaussian' in str(error), str(error)
        else:
            raise AssertionError('Unsupported backend likelihood gradient was accepted')
        for configuration, message in ((least_squares, 'BarlowBeestonLiteGaussian'),
                                       (unsupported, 'Unsupported CPU backend gradient dial')):
            try:
                build(configuration, False, False, backend='CPU', analytic=True)
            except RuntimeError as error:
                assert message in str(error), str(error)
            else:
                raise AssertionError('Incompatible analytic backend configuration initialized')


    print(f'SUCCESS: Gaussian BB CPU gradients, fit coordinates, eigenvectors, zero products and freshness ({threads} threads)')


def main():
    for threads in (1, 4):
        check_threads(threads)


if __name__ == '__main__':
    main()
