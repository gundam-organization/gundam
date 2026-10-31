#!/usr/bin/env python3
"""CPU composite programs preserve the standard engine's event weights exactly."""
import copy
import json
import math
import struct
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)
helpers = import_module('230CompositeDials')


def compare(path, dial, exact=True, backendType='CPU', backendConfig=None):
    engines = []
    for backend in (False, True):
        config = helpers.make_config(path, copy.deepcopy(dial))
        config['likelihoodInterfaceConfig']['backendManagerConfig'] = {'isEnabled': backend, 'useBackend': backendType,
            'options': [{'name': backendType, **(backendConfig or {})}]}
        engine = helpers.build(config)
        if backend and backendType == 'MPS':
            assert json.loads(engine.getBackendsManager().getDiagnosticMps())['path'] == 'eventPropagation'
        engines.append(engine)
    for p, q in [(.5, .25), (.5, .25), (-.5, .25), (-.5, .75), (1.2, -.25), (4., 2.), (.5, .25)]:
        states = []
        for engine in engines:
            likelihood = engine.getLikelihoodInterface()
            model = likelihood.getModelPropagator()
            parameters = model.getParametersManager().getParameterSetsList()[0].getParameterList()
            parameters[0].setParameterValue(p)
            parameters[1].setParameterValue(q)
            engine.evaluateLikelihood()
            sample = model.getSampleSet().getSampleList()[0]
            states.append([(event.getEventWeight(), event.getIndices().bin,
                            event.getVariables().fetchVariable('X').getVarAsDouble() if 'applyToVariable' in dial else 0.)
                           for event in sample.getEventList()])
        for reference, actual in zip(*states):
            if exact:
                assert all(struct.pack('d', a) == struct.pack('d', b) for a, b in zip(reference, actual)), (p, q, reference, actual)
            else:
                assert reference[1] == actual[1], (reference, actual)
                assert all(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12)
                           for a, b in zip(reference, actual)), (p, q, reference, actual)


def main():
    import GUNDAM
    if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        print('SKIP: backend support was not compiled.')
        return
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / '231CpuCompositeDials.root'
    helpers.write_input(path)
    p, q = {'parameterName': 'p'}, {'parameterName': 'q'}
    upstream = {'dialResponse': 'source'}
    binning = {'binningDefinition': [{'name': 'X', 'edges': [0, 1, 2, 3]}]}
    for threads in (1, 3):
        GUNDAM.setNumberOfThreads(threads)
        source = dict(name='source', dialType='Norm', dialInputList=[p],
                      useMirrorDial=True, mirrorLowEdge=-1., mirrorHighEdge=2.,
                      dialResponseLimits={'min': -.2, 'max': 1.8},
                      enableResponseSmoothing=True, dialResponseSmoothingWidth=.1)
        leaves = [dict(name='curve', dialType='Norm', dialInputList=[upstream])]
        for kind, branch in [('Graph', 'graph'), ('Spline', 'graph'), ('Spline', 'flat')]:
            leaves.append(dict(name='curve', dialType=kind, dialBranchData=branch, dialInputList=[upstream]))
        for mode in ('Bilinear', 'Bicubic'):
            leaves.append(dict(name='curve', dialType='Surface', options=mode, binning=binning,
                               dialsFilePath=str(path), dialsList='surfaces', dialInputList=[upstream, q]))
        for leaf in leaves:
            nested = helpers.composite([source, leaf], 'curve', dialResponseLimits=[0., 1.6])
            nested['name'] = 'nested'
            graph = helpers.composite([nested], 'nested', dialResponseLimits=[0., 1.4])
            compare(path, graph)
            graph.update(applyToVariable='X', variableResponseMode='Override')
            compare(path, graph)

        event = {'eventVariable': 'q3'}
        # Event-only composites have no root parameters, including when nested.
        event_only = helpers.composite([helpers.formula('event', '1+.001*[arg0]', event)], 'event')
        compare(path, event_only, exact=False)
        event_only['name'] = 'nested'
        compare(path, helpers.composite([event_only], 'nested'), exact=False)

        # Reproduce the parameter/event abscissa feeding a binned spline.
        effective = dict(name='effectiveEb', dialType='Formula', dialInputList=[
            {'name': 'q3', 'eventVariable': 'q3'},
            {'name': 'delta', 'parameterName': 'p'},
            {'name': 'alpha', 'parameterName': 'q'}],
            options={'formulaStr': '[delta]+[alpha]*(.056*[q3]-41.6)'},
            minDialResponse=-40., maxDialResponse=40.)
        curve = dict(name='curve', dialType='Spline', binning=binning,
                     dialsFilePath=str(path), dialsList='wideCurves',
                     dialInputList=[{'dialResponse': 'effectiveEb'}])
        compare(path, helpers.composite([effective, curve], 'curve'), exact=False)

        # Event values and upstream responses retain separate input order and mirrors.
        mixed = helpers.formula('sum', '2+sin([arg0])+.001*[arg1]+[arg2]^2',
                                {'dialResponse': 'source'}, event, q,
                                useMirrorDial=True, mirrorLowEdge=-1., mirrorHighEdge=2.,
                                dialResponseLimits=[1., 4.], enableResponseSmoothing=True,
                                dialResponseSmoothingWidth=.1)
        nested = helpers.composite([mixed, source], 'sum', dialResponseLimits=[0., 3.5])
        nested['name'] = 'nested'
        compare(path, helpers.composite([nested], 'nested'), exact=False)
        # Snapshot loaded X once even when the result overrides that observable.
        loaded = helpers.formula('loaded', '[arg0]+[arg1]', {'eventVariable': 'X'}, p)
        compare(path, helpers.composite([loaded], 'loaded', applyToVariable='X',
                                       variableResponseMode='Override'), exact=False)
        # Binning constants and formulas with parameter inputs only.
        binned = helpers.formula('binned', '[X]+sin([arg0])', p, binning=binning)
        compare(path, helpers.composite([binned], 'binned'), exact=False)
    print('SUCCESS: CPU composite weights and observables match, including event formulas.')


if __name__ == '__main__':
    main()
