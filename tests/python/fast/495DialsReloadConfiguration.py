#!/usr/bin/env python3
"""Loading preserves static dials; only effective overrides rebuild them."""
import json
import math
from importlib import import_module
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
fixture = import_module('490DialsBinnedBranch')


def exercise(path, mode):
    import GUNDAM
    GUNDAM.setLightOutputMode(True)
    binning = fixture.binning_definition()
    model = {'tree': 'events', 'filePathList': [str(path)]}
    if mode != 'none':
        model['overridePropagatorConfig'] = {'globalEventReweightCap': 100 if mode == 'same' else 200}
    config = {
        'enablePca': False, 'scaleParStepWithChi2Response': False,
        'generateSamplePlots': False, 'generateOneSigmaPlots': False,
        'likelihoodInterfaceConfig': {
            'jointProbabilityConfig': {'type': 'LeastSquares', 'lsqPoissonianApproximation': True},
            'dataSetList': [{'name': 'input', 'model': model}],
            'propagatorConfig': {
                'globalEventReweightCap': 100,
                'sampleSetConfig': {'sampleList': [{'name': 'sample', 'binning': binning}]},
                'parametersManagerConfig': {'parameterSetList': [{
                    'name': 'shape', 'isEnabled': True, 'numberOfParameters': 1, 'nominalStepSize': .1,
                    'parameterDefinitions': [{'name': 'p', 'priorValue': .5, 'priorType': 'Flat'}],
                    'dialSetDefinitions': [{
                        'dialType': 'Graph', 'dialInputList': [{'name': 'p'}],
                        'binning': binning, 'binnedDialsPath': f'{path}:curves:response',
                        'minDialResponse': 0.,
                    }],
                }]},
            },
        },
    }
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    likelihood = engine.getLikelihoodInterface()
    likelihood.initialize()
    propagator = likelihood.getModelPropagator()
    parameter = propagator.getParametersManager().getParameterSetsList()[0].getParameterList()[0]
    for value in (.5, 0., 1.5):
        parameter.setParameterValue(value)
        likelihood.propagateAndEvalLikelihood()
        actual = [b.sumWeights for b in propagator.getSampleSet().getSampleList()[0]
                  .getHistogram().getBinContentList()]
        expected = [2 + .2*(i + 1)*value for i in range(2)]
        assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(actual, expected)), actual
    parameter.setParameterValue(.5)
    GUNDAM.flushOutput()


def main():
    path = Path.cwd() / '495DialsReloadConfiguration.root'
    fixture.write_input(path)
    for mode, count in (('none', 1), ('same', 1), ('changed', 2)):
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                 '--exercise', str(path), mode], capture_output=True, text=True)
        log = result.stdout + result.stderr
        Path(f'495DialsReloadConfiguration-{mode}.log').write_text(log)
        assert result.returncode == 0, log
        assert log.count('Defining binned dials for') == count, log
        assert log.count('key "minDialResponse" is deprecated') == count, log
        assert log.count('Re-configuring the propagator') == count - 1, log
    print('SUCCESS: unchanged configurations preserve dials during model and data loading.')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--exercise':
        exercise(Path(sys.argv[2]), sys.argv[3])
    else:
        main()
