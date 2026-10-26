#!/usr/bin/env python3
"""Check the original fit monitor's total duration and modular stage columns."""
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


def fit():
    import GUNDAM
    import numpy as np
    import uproot

    GUNDAM.setNumberOfThreads(1)
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / '234ModularFitMonitor-input.root'
    with uproot.recreate(path) as saved:
        tree = saved.mktree('events', {'X': 'float64'})
        tree.extend({'X': np.full(20, .5)})
    config = {
        'enablePca': False, 'scaleParStepWithChi2Response': False,
        'generateSamplePlots': False, 'generateOneSigmaPlots': False,
        'minimizerConfig': {'print_level': 0, 'enablePostFitErrorFit': False,
                            'monitorRefreshRateInMs': 0},
        'likelihoodInterfaceConfig': {
            'jointProbabilityConfig': {'type': 'LeastSquares', 'lsqPoissonianApproximation': True},
            'datasetList': [{'name': 'input', 'model': {
                'tree': 'events', 'filePathList': [str(path)]}}],
            'propagatorConfig': {
                'sampleSetConfig': {'sampleList': [{'name': 'sample', 'binning': {
                    'binningDefinition': [{'name': 'X', 'edges': [0, 1]}]}}]},
                'parametersManagerConfig': {'parameterSetList': [{
                    'name': 'normalization', 'isEnabled': True, 'nominalStepSize': .1,
                    'parameterDefinitions': [{'name': 'scale', 'priorValue': 1., 'stdDev': .2,
                        'dialSetDefinitions': [{'dialType': 'Normalization'}]}]}]}}},
    }
    engine = GUNDAM.FitterEngine()
    engine.setConfig(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.configure()
    engine.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    app = GUNDAM.GundamApp('Modular fit monitor regression')
    app.openOutputFile(str(Path.cwd() / '234ModularFitMonitor-output.root'))
    engine.setSaveDir(app, 'FitterEngine')
    engine.initialize()
    engine.fit()
    assert math.isclose(engine.getLikelihoodInterface().getLastLikelihood(), 0., abs_tol=1e-6)


def main():
    if '--worker' in sys.argv:
        fit()
        return
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end='')
    if result.returncode:
        raise RuntimeError(f'Fit failed with exit code {result.returncode}')
    output = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)
    headers = [line for line in output.splitlines() if '│' in line and 'Propagation + LLH' in line]
    values = [line for line in output.splitlines() if '│ Speed' in line]
    assert headers and len(headers) == len(values), 'Missing monitor tables'
    for header, row in zip(headers, values):
        names, cells = header.split('│')[1:-1], row.split('│')[1:-1]
        assert len(names) == len(cells), (header, row)
        assert 'Propagator' in header and 'Dial Update' in header and 'Minuit2' in header, header
        assert ('Re-weight' in header and 'histograms fill' in header) or (
            'Cache::Fill' in header and 'Pull from device' in header), header
        duration = cells[next(i for i, name in enumerate(names) if name.strip() == 'Propagation + LLH')].strip()
        assert re.fullmatch(r'[0-9.]+(?:us|ms|s)', duration), duration
        assert float(re.match(r'[0-9.]+', duration)[0]) > 0, duration
    print('SUCCESS: original fit monitor displays a measured total and aligned stage columns')


if __name__ == '__main__':
    main()
