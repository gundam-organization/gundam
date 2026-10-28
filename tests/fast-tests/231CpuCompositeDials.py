#!/usr/bin/env python3
"""CPU composite programs preserve the standard engine's event weights exactly."""
import copy
import struct
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)
helpers = import_module('230CompositeDials')


def compare(path, dial):
    engines = []
    for backend in (False, True):
        config = helpers.make_config(path, copy.deepcopy(dial))
        config['likelihoodInterfaceConfig']['backendManagerConfig'] = {'isEnabled': backend, 'type': 'CPU'}
        engines.append(helpers.build(config))
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
            assert all(struct.pack('d', a) == struct.pack('d', b) for a, b in zip(reference, actual)), (p, q, reference, actual)


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
    print('SUCCESS: CPU composite programs match standard weights and observables bit for bit.')


if __name__ == '__main__':
    main()
