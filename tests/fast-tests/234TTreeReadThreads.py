#!/usr/bin/env python3
"""Check reader limits, probe candidates and friend input sizes."""
import json
import re
import subprocess
import sys
from array import array
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)


def load(payload):
    import GUNDAM

    GUNDAM.setNumberOfThreads(payload['threads'])
    GUNDAM.setLightOutputMode(True)
    config = {'likelihoodInterfaceConfig': {
        'jointProbabilityConfig': {'type': 'LeastSquares', 'lsqPoissonianApproximation': True},
        'dataSetList': [payload['dataset']],
        'propagatorConfig': {'sampleSetConfig': {'sampleList': [{
            'name': 'sample', 'binning': {
                'binningDefinition': [{'name': 'X', 'edges': [0, 1]}]}}]}}}}
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.initialize()
    sample = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()[0]
    assert len(sample.getEventList()) == payload['events']
    assert sample.getHistogram().getBinContentList()[0].sumWeights == payload['events']
    GUNDAM.flushOutput()


def main():
    import uproot

    paths = [Path.cwd() / f'234TTreeReadThreads-{i}.root' for i in range(2)]

    def check(threads, limit, candidates=(), entries=None, events=8, weight=None, small=False):
        model = {'tree': 'events', 'filePathList': entries or list(map(str, paths))}
        if weight is not None:
            model['nominalWeightFormula'] = weight
        payload = {'threads': threads, 'events': events, 'dataset': {
            'name': 'inputs', 'nbMaxThreadsForLoad': limit, 'model': model}}
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                 '--load', json.dumps(payload)], capture_output=True, text=True)
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert ('Benchmarking raw input I/O' in output) == bool(candidates), output
        if not candidates:
            if small:
                assert '(< 500 MB): using two readers' in output, output
            else:
                assert f'Using {min(threads, limit)} reader(s)' in output, output
            return
        trials = re.findall(r'Raw I/O probe: (\d+) reader\(s\), (\d+) bytes', output)
        attempts = re.findall(r'Raw I/O probe attempt (\d+)/4', output)
        assert 1 <= len(attempts) <= 4, output
        assert list(map(int, attempts)) == list(range(1, len(attempts) + 1)), output
        assert len(trials) == len(attempts) * len(candidates), output
        assert output.count('Retrying in different file regions') == len(attempts) - 1, output
        for index in range(0, len(trials), len(candidates)):
            batch = trials[index:index + len(candidates)]
            assert [int(n) for n, _ in batch] == list(candidates), output
            volumes = [int(v) for _, v in batch]
            assert max(volumes) - min(volumes) <= 2 * max(candidates), output
        selected = re.search(r'Raw I/O probe selected (\d+) reader', output)
        assert selected and int(selected[1]) in candidates, output

    try:
        for path in paths:
            with uproot.recreate(path) as output:
                tree = output.mktree('events', {'X': 'float64'})
                tree.extend({'X': array('d', [.5] * 4)})
        for threads, limit in ((1, 4), (2, 4), (4, 1), (4, 2)):
            check(threads, limit)
        check(4, 4, small=True)
        # Sparse padding tests the size threshold without writing 500 MB.
        for path in paths:
            with path.open('r+b') as output:
                output.truncate(250_000_000)
        for threads, limit in ((1, 4), (2, 4), (4, 1), (4, 2)):
            check(threads, limit)
        check(3, 4, (2, 3))
        check(4, 4, (2, 3, 4))
        check(4, 3, (2, 3))
        check(6, 6, (2, 3, 4, 6))
        main = {'name': 'main', 'path': str(paths[0]), 'friendList': [
            {'name': 'weights', 'path': str(paths[1]) + ':events'}]}
        check(4, 4, (2, 3, 4), [main], events=4, weight='2 * weights.X')
        main['friendList'][0]['isEnabled'] = False
        check(4, 4, entries=[main], events=4, small=True)
        main['friendList'] = [{'name': alias, 'path': str(paths[0]) + ':events'}
                              for alias in ('first', 'second')]
        check(4, 4, entries=[main], events=4, small=True)
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
    print('SUCCESS: reader limits, minimum of two readers, bounded probes and friends preserve predictions.')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--load':
        load(json.loads(sys.argv[2]))
    else:
        main()
