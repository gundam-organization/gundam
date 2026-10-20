#!/usr/bin/env python3
"""Exercise benchmark workloads, deterministic comparisons and eigen coordinates."""
import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)


def main():
    import GUNDAM
    import numpy as np
    import uproot
    if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        print('SKIP: GUNDAM was built without backends')
        return
    base = Path.cwd() / '233CpuWeightBenchmarkScenarios'
    base.mkdir(exist_ok=True)
    root = base / 'input.root'
    with uproot.recreate(root) as saved:
        tree = saved.mktree('tree_mc', {'X': 'float64'})
        tree.extend({'X': np.tile([-.5, .5], 90)})
    benchmark = Path(__file__).resolve().parents[1] / 'regular-tests' / '222CpuWeightSchedulingBenchmark.py'
    for eigen in (False, True):
        definitions = [dict(name=f'p{i}', priorValue=1., stdDev=.1 * (i + 1),
                            dialSetDefinitions=[{'dialType': 'Normalization'}]) for i in range(3)]
        definitions += [dict(name='fixed', priorValue=1., stdDev=.1, isFixed=True,
                             dialSetDefinitions=[{'dialType': 'Normalization'}])]
        if not eigen:
            definitions[0]['parameterLimits'] = [0.999, 1.001]
        config = {'fitterEngineConfig': {'likelihoodInterfaceConfig': {
            'jointProbabilityConfig': {'type': 'PoissonLLH'},
            'dataSetList': [{'name': 'sample', 'model': {'tree': 'tree_mc', 'filePathList': [str(root)]}}],
            'propagatorConfig': {
                'sampleSetConfig': {'sampleList': [{'name': 'X', 'dataSets': ['sample'],
                    'binning': {'binningDefinition': [{'name': 'X', 'edges': [-1, 0, 1]}]}}]},
                'parametersManagerConfig': {'parameterSetList': [{'name': 'test', 'isEnabled': True,
                    'enableEigenDecomp': eigen, 'parameterDefinitions': definitions}]}}}}}
        path = base / f'config-{eigen}.json'
        path.write_text(json.dumps(config))
        output = base / f'results-{eigen}-{os.getpid()}'
        with (base / f'benchmark-{eigen}.log').open('w') as log:
            subprocess.run([sys.executable, str(benchmark), '--config', str(path), '--threads', '4',
                            '--factors', '2', '5', '10', '--calls', '7', '--warmup', '3',
                            '--trials', '2', '--rounds', '2', '--output', str(output)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((output / 'summary.json').read_text())
        assert len(summary) == 32
        assert len({r['pid'] for r in summary}) == 1  # Same loaded engine for all workloads/layouts.
        manifest = json.loads((output / 'parameters-t4.json').read_text())
        # The fit uses four eigen coordinates here; an original-space fixed
        # flag does not imply one particular eigen coordinate is fixed.
        assert len(manifest) == (4 if eigen else 3)
        assert {p['space'] for p in manifest} == {'eigen' if eigen else 'original'}
        signatures = {}
        for result in summary:
            scenario = result['scenario']
            assert result['histogramsAgree'] and result['likelihoodsAgree']
            assert result['weightSpeedupVsFixed'] > 0 and result['evaluationSpeedupVsFixed'] > 0
            expected_count = {'constant': 0, 'local1': 1, 'local2': 2, 'global': len(manifest)}[scenario]
            for trial in result['trials']:
                assert trial['changedFitCoordinates'] == [expected_count]
            signature = [t['sequenceDigest'] for t in result['trials']]
            assert signature == signatures.setdefault(scenario, signature)
        constant_histogram = json.loads((output / 'round1-t4-constant-Fixed.json').read_text())['histogram']
        for path in output.glob('round*.json'):
            result = json.loads(path.read_text())
            if result['scenario'] != 'constant':
                assert result['histogram'] != constant_histogram
            for raw in result['raw']:
                assert len(raw['likelihoods']) == 7
                spread = max(raw['likelihoods']) - min(raw['likelihoods'])
                assert spread == 0 if result['scenario'] == 'constant' else spread > 1e-12
        assert not list(output.glob('*.partial.json'))
    print('SUCCESS: all benchmark scenarios vary the intended coordinates, replay identical sequences, and preserve predictions across layouts.')


if __name__ == '__main__':
    main()
