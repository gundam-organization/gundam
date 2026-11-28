#!/usr/bin/env python3
"""Detect a flat likelihood direction alongside a curved parameter."""
import json
from pathlib import Path

import GUNDAM
import numpy as np
import uproot


def main():
    GUNDAM.setNumberOfThreads(1)
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / '550NumericsFlatLikelihoodDirections.root'
    with uproot.recreate(path) as output:
        tree = output.mktree('events', {'X': 'float64'})
        tree.extend({'X': np.full(10, .5, dtype=np.float64)})
    parameters = [
        {'name': 'curved', 'priorValue': 1., 'priorType': 'Flat',
         'dialSetDefinitions': [{'dialType': 'Normalization'}]},
        {'name': 'plateau', 'priorValue': 2., 'priorType': 'Flat',
         'parameterLimits': [2., 5.],
         'dialSetDefinitions': [{'dialType': 'Normalization', 'maxDialResponse': 1.}]},
    ]
    li = {
        'jointProbabilityConfig': {'type': 'PoissonLLH'},
        'datasetList': [{'name': 'input', 'model': {'tree': 'events', 'filePathList': [str(path)]}}],
        'propagatorConfig': {
            'sampleSetConfig': {'sampleList': [{'name': 'sample', 'binning': {
                'binningDefinition': [{'name': 'X', 'edges': [0, 1]}]}}]},
            'parametersManagerConfig': {'parameterSetList': [{'name': 'response', 'isEnabled': True,
                'nominalStepSize': .1, 'parameterDefinitions': parameters}]}}}
    if hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        li['backendManagerConfig'] = {'isEnabled': True, 'type': 'CPU'}
    config = {'likelihoodInterfaceConfig': li, 'enablePca': False,
              'scaleParStepWithChi2Response': False, 'generateSamplePlots': False,
              'generateOneSigmaPlots': False,
              'minimizerConfig': {'enablePostFitErrorFit': False, 'print_level': 0,
                                  'useNormalizedFitSpace': False}}
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.initialize()
    assert engine.getMinimizer().checkFlatDirections() == 1, 'Expected exactly one flat direction'
    print('SUCCESS: detected the flat likelihood direction')


if __name__ == '__main__':
    main()
