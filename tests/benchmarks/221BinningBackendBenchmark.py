#!/usr/bin/env python3
"""Benchmark complete likelihood evaluations with CPU/Metal migration and smoothing."""
import argparse
import json
import math
import statistics
import sys
import time
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fast-tests'))
import_module('005PythonSetup-utils').requireVenv(__file__)


def make_engine(path, bins, mode, backend):
    import GUNDAM
    import numpy as np
    sample = {'name': 'sample', 'binning': {'binningDefinition': [
        {'name': 'X', 'edges': np.linspace(0, 64, bins + 1).tolist()}]}}
    if mode != 'hard':
        width = {'width': .2} if mode == 'absolute' else {'widthFraction': .3}
        sample['binningResponse'] = {'isEnabled': True, 'variables': [{'name': 'X', **width}]}
    li = {
        'jointProbabilityConfig': {'type': 'LeastSquares', 'lsqPoissonianApproximation': True},
        'dataSetList': [{'name': 'inputs', 'model': {'tree': 'events', 'filePathList': [str(path)],
                                                     'nominalWeightFormula': 'weight'}}],
        'propagatorConfig': {
            'sampleSetConfig': {'sampleList': [sample]},
            'parametersManagerConfig': {'parameterSetList': [{
                'name': 'parameters', 'isEnabled': True, 'nominalStepSize': .1,
                'parameterDefinitions': [
                    {'name': 'scale', 'priorValue': 1., 'priorType': 'Flat',
                     'dialSetDefinitions': [{'dialType': 'Normalization', 'applyToVariable': 'X'}]},
                    {'name': 'weight', 'priorValue': 1., 'priorType': 'Flat',
                     'dialSetDefinitions': [{'dialType': 'Normalization'}]},
                ]}]},
        },
    }
    if backend:
        li['backendManagerConfig'] = {'isEnabled': True, 'type': backend}
    config = {'likelihoodInterfaceConfig': li, 'enablePca': False, 'scaleParStepWithChi2Response': False,
              'generateSamplePlots': False, 'generateOneSigmaPlots': False}
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.initialize()
    if backend:
        engine.getBackendsManager().setEnableAutoMaterialize(False)
    return engine


def main():
    import GUNDAM
    import numpy as np
    import uproot
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=int, default=100000)
    parser.add_argument('--bins', type=int, nargs='+', default=[64, 256])
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=9)
    parser.add_argument('--output', type=Path,
                        default=Path(f'benchmark-{Path(__file__).stem}-{time.strftime("%Y%m%d%H%M%S")}'),
                        help='Output directory (default: ./benchmark-<script-name>-YYYYMMDDHHmmss); a .json file is also accepted')
    parser.add_argument('--backends', nargs='+', choices=['CPU', 'MPS'],
                        default=['CPU', 'MPS'] if sys.platform == 'darwin' else ['CPU'])
    args = parser.parse_args()
    if min([args.events, args.threads, args.repeat] + args.bins) <= 0:
        parser.error('Counts must be positive')
    args.output = args.output.resolve()
    if args.output.suffix.lower() == '.json':
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report_path = args.output
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        report_path = args.output / 'summary.json'
    print(f'Results: {report_path.parent}', flush=True)
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(args.threads)
    path = report_path.parent / '221BinningBackendBenchmark.root'
    random = np.random.default_rng(724)
    with uproot.recreate(path) as output:
        tree = output.mktree('events', {'X': 'float64', 'weight': 'float64'})
        tree.extend({'X': random.uniform(-1, 65, args.events), 'weight': random.uniform(.5, 2., args.events)})
    rows = []
    for bins in args.bins:
        for mode in ('hard', 'absolute', 'relative'):
            reference = None
            for backend in (None, *args.backends):
                engine = make_engine(path, bins, mode, backend)
                li = engine.getLikelihoodInterface()
                parameters = li.getModelPropagator().getParametersManager().getParameterSetsList()[0].getParameterList()
                row = {'events': args.events, 'bins': bins, 'threads': args.threads,
                       'mode': mode, 'backend': backend or 'legacy'}
                for change in ('migration', 'weight'):
                    durations = []
                    for iteration in range(args.repeat + 3):
                        parameters[0 if change == 'migration' else 1].setParameterValue(1. + .02 * (iteration % 2))
                        start = time.perf_counter()
                        engine.evaluateLikelihood()
                        elapsed = time.perf_counter() - start
                        if iteration >= 3:
                            durations.append(elapsed)
                    row[change + '_ms'] = statistics.median(durations) * 1000
                    if backend:
                        timing = engine.getBackendsManager().getLastTimingSummary()
                        row[change + '_rebinned_events'] = timing.rebinnedEventCount
                        row[change + '_candidates'] = timing.binCandidateCount
                        row[change + '_device_ms'] = timing.devicePropagationSeconds * 1000
                        row[change + '_state_readback_bytes'] = timing.eventStateReadbackBytes
                        row['contribution_buffer_bytes'] = timing.contributionBufferBytes
                        if backend == 'MPS':
                            assert timing.eventStateReadbackBytes == 0
                            assert timing.eventWeightReadbackBytes == 0
                        for stage in ('variable', 'binning', 'eventWeights', 'histogram'):
                            row[change + '_' + stage + '_ms'] = getattr(timing, stage + 'StageSeconds') * 1000
                if backend:
                    engine.getBackendsManager().materialize('Histograms')
                sample = li.getModelPropagator().getSampleSet().getSampleList()[0]
                values = [(content.sumWeights, content.sqrtSumSqWeights) for content in sample.getHistogram().getBinContentList()]
                if reference is None:
                    reference = values
                else:
                    for actual, expected in zip(values, reference):
                        for a, b in zip(actual, expected):
                            assert math.isclose(a, b, rel_tol=2e-5 if backend == 'MPS' else 1e-10, abs_tol=2e-5 if backend == 'MPS' else 1e-10), (a, b)
                rows.append(row)
                print(json.dumps(row), flush=True)
    report_path.write_text(json.dumps(rows, indent=2) + '\n')
    print('SUCCESS: benchmark histogram sums and errors agree')


if __name__ == '__main__':
    main()
