#!/usr/bin/env python3
"""Flat-direction warnings, boundary probes, restoration, and lifecycle hooks."""
import json
import math
import re
import subprocess
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)


def child(mode):
    import GUNDAM
    import numpy as np
    import uproot
    GUNDAM.setNumberOfThreads(1)
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / f'225FlatLikelihoodDirections-{mode}.root'
    with uproot.recreate(path) as output:
        tree = output.mktree('events', {'X': 'float64'})
        tree.extend({'X': np.full(10, .5, dtype=np.float64)})
    parameters = [{'name': 'curved', 'priorValue': 1., 'priorType': 'Flat',
                   'dialSetDefinitions': [{'dialType': 'Normalization'}]}]
    if mode in ('dry', 'blocked', 'probe'):
        for name, fixed in [('plateau', False), ('fixedPlateau', True)]:
            parameters.append({'name': name, 'priorValue': 2., 'priorType': 'Flat',
                               'isFixed': fixed, 'parameterLimits': [2., 5.],
                               'dialSetDefinitions': [{'dialType': 'Normalization', 'maxDialResponse': 1.}]})
    li = {
        'jointProbabilityConfig': {'type': 'PoissonLLH'},
        'datasetList': [{'name': 'input', 'model': {'tree': 'events', 'filePathList': [str(path)]}}],
        'propagatorConfig': {
            'sampleSetConfig': {'sampleList': [{'name': 'sample', 'binning': {
                'binningDefinition': [{'name': 'X', 'edges': [0, 1]}]}}]},
            'parametersManagerConfig': {'parameterSetList': [{'name': 'response', 'isEnabled': True,
                'nominalStepSize': .1, 'parameterDefinitions': parameters}]}}}
    config = {'likelihoodInterfaceConfig': li, 'enablePca': False,
              'scaleParStepWithChi2Response': False, 'generateSamplePlots': False,
              'generateOneSigmaPlots': False,
              'minimizerConfig': {'enablePostFitErrorFit': mode == 'fit', 'print_level': 0,
                                  'useNormalizedFitSpace': False}}
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.initialize()
    engine.setIsDryRun(mode in ('dry', 'dryClean'))
    llh = engine.getLikelihoodInterface()
    model = llh.getModelPropagator()
    curved = model.getParametersManager().getParameterSetsList()[0].getParameterList()[0]
    if mode == 'restore':
        curved.setParameterValue(1.3)
        llh.propagateAndEvalLikelihood()
        before = llh.getLastLikelihood()
        data = [b.sumWeights for b in llh.getDataPropagator().getSampleSet().getSampleList()[0].getHistogram().getBinContentList()]
        assert engine.getMinimizer().checkFlatDirections() == 0
        assert curved.getParameterValue() == 1.3
        assert math.isclose(llh.getLastLikelihood(), before, abs_tol=1e-12)
        assert data == [b.sumWeights for b in llh.getDataPropagator().getSampleSet().getSampleList()[0].getHistogram().getBinContentList()]
    elif mode == 'probe':
        assert engine.getMinimizer().checkFlatDirections() == 1
    else:
        try:
            engine.fit()
        except RuntimeError as error:
            assert mode in ('dry', 'blocked'), str(error)
            assert 'Cannot start fit' in str(error), str(error)
            print('FIT_BLOCKED', mode, flush=True)
        else:
            assert mode not in ('dry', 'blocked'), 'A flat fit must be rejected'
        if mode in ('dry', 'blocked'):
            assert curved.getParameterValue() == 1.
            assert model.getParametersManager().getParameterSetsList()[0].getParameterList()[1].getParameterValue() == 2.
    print('CHILD_OK', mode, flush=True)


def main():
    for mode in ('dry', 'blocked', 'dryClean', 'fit', 'restore', 'probe'):
        result = subprocess.run([sys.executable, __file__, mode], capture_output=True, text=True, errors='replace')
        log = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', result.stdout + result.stderr)
        Path(f'225FlatLikelihoodDirections-{mode}.log').write_text(log)
        assert result.returncode == 0 and f'CHILD_OK {mode}' in log, log[-8000:]
        assert log.count('Checking LLH flat directions...') == (2 if mode == 'fit' else 1), log[-8000:]
        flat = [line for line in log.splitlines() if 'Flat LLH direction:' in line]
        if mode in ('dry', 'blocked', 'probe'):
            assert len(flat) == 1 and 'ERROR' in flat[0] and 'response/#1_plateau' in flat[0], flat
            assert 'value = 2' in flat[0] and 'first derivative = 0' in flat[0] and 'second derivative = 0' in flat[0], flat
            assert 'Minimization ended' not in log
            if mode != 'probe':
                assert f'FIT_BLOCKED {mode}' in log
        else:
            assert not flat, flat  # A zero gradient at a curved minimum is valid.
    print('SUCCESS: flat fits are blocked, dry-run/error hooks run, standalone checks stay informative, and state is restored')


if __name__ == '__main__':
    modes = ('dry', 'blocked', 'dryClean', 'fit', 'restore', 'probe')
    child(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] in modes else main()
