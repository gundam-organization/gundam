#!/usr/bin/env python3
"""CPU formula bytecode matches ROOT through configuration, binding and caching."""
import copy
import json
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)
helpers = import_module('231EventVarInputForFormulaDials')


def compare(path, dial, backendType='CPU', backendConfig=None):
    engines = []
    for enabled in (False, True):
        configuration = helpers.config(path, copy.deepcopy(dial))
        configuration['likelihoodInterfaceConfig']['backendManagerConfig'] = {
            'isEnabled': enabled, 'type': backendType, 'backendConfig': backendConfig or {}}
        engine = helpers.build(configuration)
        if enabled and backendType == 'MPS':
            assert json.loads(engine.getBackendsManager().getDiagnosticMps())['path'] == 'eventPropagation'
        engines.append(engine)
    for delta, alpha in [(.5, .25), (.5, .25), (-.5, .25), (-.5, .75), (0., 1.), (3., 2.), (.5, .25)]:
        states = []
        for engine in engines:
            likelihood = engine.getLikelihoodInterface()
            model = likelihood.getModelPropagator()
            parameters = model.getParametersManager().getParameterSetsList()[0].getParameterList()
            parameters[0].setParameterValue(delta)
            parameters[1].setParameterValue(alpha)
            likelihood.propagateAndEvalLikelihood()
            sample = model.getSampleSet().getSampleList()[0]
            states.append(([
                (event.getEventWeight(), event.getVariables().fetchVariable('X').getVarAsDouble()
                 if 'applyToVariable' in dial else 0.)
                for event in sample.getEventList()],
                [content.sumWeights for content in sample.getHistogram().getBinContentList()]))
        for reference, actual in zip(states[0][0], states[1][0]):
            for expected, value in zip(reference, actual):
                helpers.close(value, expected)
        for expected, value in zip(states[0][1], states[1][1]):
            helpers.close(value, expected)


def main():
    import GUNDAM
    import numpy as np
    import uproot
    if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        print('SKIP: backend support was not compiled.')
        return
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / '232CpuFormulaDials.root'
    with uproot.recreate(path) as output:
        tree = output.mktree('events', {'X': 'float64', 'rawQ3': 'float64'})
        tree.extend({'X': .25 + .5*np.arange(6), 'rawQ3': 250.*np.arange(6)})
    delta = {'name': 'offset', 'parameterName': 'delta'}
    alpha = {'name': 'slope', 'parameterName': 'alpha'}
    q3 = {'name': 'momentum', 'eventVariable': 'q3'}
    binning = {'binningDefinition': [{'name': 'X', 'edges': [0, 1, 2, 3]}]}
    cases = [
        helpers.formula('100+[offset]+[slope]*(.056*[momentum]-41.6)', [q3, alpha, delta]),
        helpers.formula('2+sin([offset])+cos([slope])+exp(-.001*[momentum])', [delta, q3, alpha]),
        helpers.formula('1+log(1+[offset]^2)+sqrt(1+[slope]^2)', [delta, alpha]),
        helpers.formula('2+asin(.1*[offset])+atan2([offset],1+[slope]^2)', [alpha, delta]),
        helpers.formula('2+sin([offset])', [delta]),
        helpers.formula('2+sin([offset])', [delta, alpha]),  # Unreferenced declared input.
        helpers.formula('1+.001*[momentum]', [q3]),
        helpers.formula('2.5', []),
        helpers.formula('[X]+.001*[momentum]*[offset]^2', [q3, delta], binning=binning),
        helpers.formula('[X]', [], binning=binning),
        helpers.formula('[X]', [{'name': 'X', 'eventVariable': 'X'}], binning=binning),
        helpers.formula('1+.001*[momentum]+[offset]^2', [delta, q3],
                        useMirrorDial=True, mirrorLowEdge=0., mirrorHighEdge=1000.),
        helpers.formula('2+exp([offset])', [delta], minDialResponse=2.5, maxDialResponse=4.,
                        enableResponseSmoothing=True, dialResponseSmoothingWidth=.1),
        helpers.formula('2+sin([offset])', [delta], applyCondition='X < 2'),
        helpers.formula('1+.1*sin([offset])', [delta], applyToVariable='X', variableResponseMode='Override'),
    ]
    for threads in (1, 3):
        GUNDAM.setNumberOfThreads(threads)
        for dial in cases:
            compare(path, dial)
    # A valid ROOT formula outside the supported subset must fail explicitly.
    unsupported = helpers.config(path, helpers.formula('2+([offset]>0 ? 1 : 0)', [delta]))
    unsupported['likelihoodInterfaceConfig']['backendManagerConfig'] = {'isEnabled': True, 'type': 'CPU'}
    helpers.reject(unsupported, 'Formula bytecode')
    print('SUCCESS: CPU formula bytecode matches ROOT weights, histograms and observables.')


if __name__ == '__main__':
    main()
