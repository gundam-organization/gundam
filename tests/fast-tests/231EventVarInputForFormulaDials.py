#!/usr/bin/env python3
"""Native Formula dials share their definition and bind loaded event inputs."""
import copy
import json
import math
import subprocess
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)


def formula(expression, inputs, **options):
    return dict(name='eventFormula', dialType='Formula', dialInputList=inputs,
                options={'formulaStr': expression}, **options)


def config(path, dial):
    return {'enablePca': False, 'scaleParStepWithChi2Response': False,
            'generateSamplePlots': False, 'generateOneSigmaPlots': False,
            'likelihoodInterfaceConfig': {
                'jointProbabilityConfig': {'type': 'LeastSquares', 'lsqPoissonianApproximation': True},
                'dataSetList': [{'name': 'input', 'model': {'tree': 'events', 'filePathList': [str(path)],
                    'variableDict': [{'name': 'q3', 'expr': 'rawQ3 + 0.'}]}}],
                'propagatorConfig': {
                    'sampleSetConfig': {'sampleList': [{'name': 'sample', 'binning': {
                        'binningDefinition': [{'name': 'X', 'edges': [0, 1, 2, 3]}]}}]},
                    'parametersManagerConfig': {'parameterSetList': [{
                        'name': 'parameters', 'isEnabled': True, 'nominalStepSize': .1,
                        'parameterDefinitions': [dict(name=name, priorValue=prior, priorType='Flat')
                                                 for name, prior in [('delta', .5), ('alpha', .25)]],
                        'dialSetDefinitions': [dial]}]}}}}


def build(configuration):
    import GUNDAM
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(configuration))))
    engine.initialize()
    return engine


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-11), (actual, expected)


def exercise(path, dial, expected):
    engine = build(config(path, dial))
    likelihood = engine.getLikelihoodInterface()
    model = likelihood.getModelPropagator()
    parameters = model.getParametersManager().getParameterSetsList()[0].getParameterList()
    sample = model.getSampleSet().getSampleList()[0]
    points = [(.5, .25), (.5, .25), (-.5, .25), (-.5, .75), (0., 1.), (30., 2.), (.5, .25)]
    for delta, alpha in points:
        parameters[0].setParameterValue(delta)
        parameters[1].setParameterValue(alpha)
        for _ in range(2):
            likelihood.propagateAndEvalLikelihood()
            sums = [0., 0., 0.]
            for event in sample.getEventList():
                index = event.getIndices().treeEntry
                weight = expected(delta, alpha, index)
                close(event.getEventWeight(), weight)
                sums[index//2] += weight
            for content, expected_sum in zip(sample.getHistogram().getBinContentList(), sums):
                close(content.sumWeights, expected_sum)
            for event in likelihood.getDataPropagator().getSampleSet().getSampleList()[0].getEventList():
                close(event.getEventWeight(), expected(.5, .25, event.getIndices().treeEntry))


def reject(configuration, message):
    try:
        build(configuration)
    except Exception as error:
        assert message.lower() in str(error).lower(), (message, str(error))
    else:
        raise AssertionError('Invalid Formula accepted: ' + message)


def main():
    import GUNDAM
    import numpy as np
    import uproot
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd()/'231EventVarInputForFormulaDials.root'
    with uproot.recreate(path) as output:
        tree = output.mktree('events', {'X': 'float64', 'rawQ3': 'float64'})
        tree.extend({'X': .25+.5*np.arange(6), 'rawQ3': 250.*np.arange(6)})
    delta = {'name': 'delta'}
    alpha = {'name': 'alpha'}
    q3 = {'eventVariable': 'q3'}
    for threads in (1, 2):
        GUNDAM.setNumberOfThreads(threads)
        mixed = formula('2+x[1]*x[1]+x[2]*x[2]+.001*(x[0]+x[3])',
                        [{'eventVariable': '[q3]'}, delta, alpha, q3], printDialSummary=True)
        exercise(path, mixed, lambda d, a, i: 2+d*d+a*a+.5*i)
        legacy = copy.deepcopy(mixed)
        legacy['dialConfig'] = legacy.pop('options')
        exercise(path, legacy, lambda d, a, i: 2+d*d+a*a+.5*i)
        exercise(path, formula('1+.001*x[0]', [q3]), lambda d, a, i: 1+.25*i)
        exercise(path, formula('1+x[0]*x[0]', [delta]), lambda d, a, i: 1+d*d)
        # Shift the signed abscissa by 50 so this standalone dial is a positive weight.
        bounded = formula('50+x[0]+x[1]*(.056*x[2]-41.6)', [delta, alpha, q3],
                          minDialResponse=10., maxDialResponse=90.)
        exercise(path, bounded, lambda d, a, i: 50+max(-40., min(40., d+a*(.056*250*i-41.6))))
        binned = formula('[X]+.001*x[0]*x[1]*x[1]', [q3, delta], binning={
            'binningDefinition': [{'name': 'X', 'edges': [0, 1, 2, 3]}]})
        exercise(path, binned, lambda d, a, i: i//2+.5+.25*i*d*d)
        mirrored = formula('1+.001*x[0]', [q3], useMirrorDial=True, mirrorLowEdge=0., mirrorHighEdge=1000.)
        exercise(path, mirrored, lambda d, a, i: 1+.001*(250*i if i <= 4 else 2000-250*i))
        conditioned = formula('1+.001*x[0]', [q3], applyCondition='X < 2')
        exercise(path, conditioned, lambda d, a, i: 1+.25*i if i < 4 else 1.)
    reject(config(path, formula('x[0]', [{'name': 'delta', 'eventVariable': 'q3'}])), 'exactly one of')
    reject(config(path, formula('x[0]', [{'eventVariable': ''}])), 'must not be empty')
    reject(config(path, dict(dialType='Normalization', dialInputList=[q3])), 'only supported by Formula')
    reject(config(path, formula('x[2]', [q3])), 'too few inputs')
    missing = Path.cwd()/'231EventVarInputForFormulaDials-missing.json'
    missing.write_text(json.dumps(config(path, formula('x[0]', [{'eventVariable': 'missingQ3'}]))))
    result = subprocess.run([sys.executable, __file__, '--load-config', str(missing)], capture_output=True, text=True)
    assert result.returncode != 0 and 'missingQ3' in result.stdout+result.stderr, result
    print('SUCCESS: native event Formula inputs, caches, bins, mirroring, aliases and Asimov copies.')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--load-config':
        build(json.loads(Path(sys.argv[2]).read_text()))
    else:
        main()
