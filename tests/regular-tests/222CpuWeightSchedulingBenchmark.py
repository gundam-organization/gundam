#!/usr/bin/env python3
"""Compare fixed CPU ranges and several dynamic block counts on an existing input model.

Each mode loads in a separate process to keep peak memory bounded. Timed calls
repeat explicit propagations at the initialized point, after cache warmup; they
do not run a minimizer or Hesse. Results describe reweight scheduling at that
point, not whole-fit convergence or cache invalidation performance.
"""
import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True


def measure(args):
    import GUNDAM
    GUNDAM.setRuntimeWorkingDirectory(str(args.work_dir))
    GUNDAM.setLightOutputMode(True)
    threads = args.threads[0]
    GUNDAM.setNumberOfThreads(threads)
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfig(str(args.config))
    if args.override:
        builder.override(args.override)
    config = json.loads(builder.getConfig().toString())['fitterEngineConfig']
    config.update(enablePca=False, scaleParStepWithChi2Response=False,
                  generateSamplePlots=False, generateOneSigmaPlots=False)
    profile = args.result.with_suffix('.jsonl')
    config['likelihoodInterfaceConfig']['backendManagerConfig'] = {
        'isEnabled': True, 'type': 'CPU', 'backendConfig': {
            'weightScheduling': args.mode, 'profileOutputPath': str(profile),
            'profileSampleInterval': 1, 'profileMaxSamples': 5}}
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    li = engine.getLikelihoodInterface()
    li.setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    engine.initialize()
    profile = profile.with_name(f'{profile.stem}-{os.getpid()}{profile.suffix}')
    metadata = [r for line in profile.read_text().splitlines()
                if (r := json.loads(line))['kind'] == 'metadata'][-1]
    if metadata.get('weightBlocksPerThread') != args.factor:
        raise RuntimeError('Rebuild GUNDAM and select its Python module: diagnostic block factor is unsupported.')
    if args.mode == 'Dynamic' and metadata['weightScheduling'] != 'Dynamic':
        raise RuntimeError('The dataset is too small (or only one thread is used) for this dynamic block count.')
    # All five diagnostic exports finish before timing starts. No probes run
    # in the measured interval after the profiler reaches its sample cap.
    for _ in range(args.warmup):
        engine.evaluateLikelihood()
    trials, raw = [], []
    for _ in range(args.trials):
        stages, caches, histograms, complete = [], [], [], []
        for _ in range(args.calls):
            start = time.perf_counter()
            engine.evaluateLikelihood()
            complete.append(time.perf_counter() - start)
            timing = engine.getBackendsManager().getLastTimingSummary()
            stages.append(timing.eventWeightsStageSeconds)
            caches.append(timing.cachedDialStageSeconds)
            histograms.append(timing.histogramStageSeconds)
        raw.append({'weightSeconds': stages, 'cacheSeconds': caches,
                    'histogramSeconds': histograms, 'evaluationSeconds': complete})
        trials.append({'weightMedianMs': statistics.median(stages) * 1000,
                       'weightMeanMs': statistics.mean(stages) * 1000,
                       'weightP95Ms': sorted(stages)[int((len(stages) - 1) * .95)] * 1000,
                       'cacheMedianMs': statistics.median(caches) * 1000,
                       'histogramMedianMs': statistics.median(histograms) * 1000,
                       'evaluationMedianMs': statistics.median(complete) * 1000})
    histogram = [(v.sumWeights, v.sqrtSumSqWeights)
                 for sample in li.getModelPropagator().getSampleSet().getSampleList()
                 for v in sample.getHistogram().getBinContentList()]
    result = {'mode': args.mode, 'blocksPerThread': args.factor if args.mode == 'Dynamic' else None,
              'actualScheduling': metadata['weightScheduling'], 'workBlocks': metadata['weightBlockCount'],
              'threads': threads, 'host': platform.node(), 'pid': os.getpid(), 'commit': metadata['commit'],
              'pythonModule': GUNDAM.__file__, 'compilerOptimizationEnabled': metadata['compilerOptimizationEnabled'],
              'allowedCpus': metadata['system'].get('allowedCpus'), 'events': metadata['events'],
              'eventDialReferences': metadata['eventDialReferences'], 'raw': raw,
              'callsPerTrial': args.calls, 'trials': trials, 'histogram': histogram,
              'weightMedianMs': statistics.median(t['weightMedianMs'] for t in trials),
              'evaluationMedianMs': statistics.median(t['evaluationMedianMs'] for t in trials),
              'histogramMedianMs': statistics.median(t['histogramMedianMs'] for t in trials)}
    write_json(args.result, result)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--override', nargs='*', default=[])
    parser.add_argument('--work-dir', type=Path, default=Path.cwd())
    parser.add_argument('--threads', nargs='+', type=int, default=[8])
    parser.add_argument('--factors', nargs='+', type=int, default=[2, 5, 10])
    parser.add_argument('--calls', type=int, default=300)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=1,
                        help='Reload each case per round; alternate forward and reverse case order')
    parser.add_argument('--output', type=Path,
                        default=Path(time.strftime('cpu-weight-benchmark-%Y%m%d-%H%M%S-') + str(os.getpid())))
    parser.add_argument('--mode', choices=['Fixed', 'Dynamic'], help=argparse.SUPPRESS)
    parser.add_argument('--factor', type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument('--result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min([*args.threads, *args.factors, args.factor, args.calls, args.warmup, args.trials, args.rounds]) <= 0:
        parser.error('All counts must be positive')
    if args.warmup < 5:
        parser.error('--warmup must be at least 5 to finish diagnostic exports before timing')
    if 1 in args.threads:
        parser.error('Dynamic comparisons require at least two threads')
    args.work_dir = args.work_dir.resolve()
    args.config = (args.work_dir / args.config).resolve()
    if args.mode:
        measure(args)
        return
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    results, reference = [], None
    cases = [(threads, mode, factor) for threads in dict.fromkeys(args.threads)
             for mode, factor in [('Fixed', 2), *[('Dynamic', f) for f in dict.fromkeys(args.factors)]]]
    print(f'Results: {args.output}\n{len(cases) * args.rounds} sequential model loads; no simultaneous model copies.', flush=True)
    write_json(args.output / 'settings.json', {
        'config': str(args.config), 'override': args.override, 'workDir': str(args.work_dir),
        'threads': args.threads, 'factors': args.factors, 'rounds': args.rounds,
        'calls': args.calls, 'warmup': args.warmup, 'trials': args.trials, 'host': platform.node()})
    for round_index in range(args.rounds):
        ordered = cases if round_index % 2 == 0 else list(reversed(cases))
        for threads, mode, factor in ordered:
            tag = f'round{round_index + 1}-t{threads}-{mode}' + (f'-x{factor}' if mode == 'Dynamic' else '')
            path = args.output / f'{tag}.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--config', str(args.config),
                       '--work-dir', str(args.work_dir), '--threads', str(threads), '--calls', str(args.calls),
                       '--warmup', str(args.warmup), '--trials', str(args.trials), '--mode', mode,
                       '--factor', str(factor), '--result', str(path)]
            if args.override:
                command += ['--override', *args.override]
            environment = os.environ.copy()
            environment['GUNDAM_CPU_DIAGNOSTIC_BLOCKS_PER_THREAD'] = str(factor)
            print(f'Loading and measuring {tag}...', flush=True)
            with (args.output / f'{tag}.log').open('w') as log:
                subprocess.run(command, cwd=args.work_dir, stdout=log, stderr=subprocess.STDOUT,
                               check=True, env=environment)
            result = json.loads(path.read_text())
            result['round'] = round_index + 1
            values = result.pop('histogram')
            if reference is None:
                reference = values
            else:
                if len(values) != len(reference) or any(
                        not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(actual, expected))
                        for actual, expected in zip(values, reference)):
                    raise RuntimeError(f'{tag}: histogram sums/errors differ from the reference')
            result['histogramsAgree'] = True
            result.pop('raw')  # Raw call timings remain in the individual case JSON.
            results.append(result)
            for measured in results:
                baseline = next((r for r in results if r['round'] == measured['round']
                                 and r['threads'] == measured['threads'] and r['mode'] == 'Fixed'), None)
                if baseline:
                    measured['weightSpeedupVsFixed'] = baseline['weightMedianMs'] / measured['weightMedianMs']
            write_json(args.output / 'summary.json', results)
            print(f"{tag}: reweight median {result['weightMedianMs']:.3f} ms; histograms agree", flush=True)
    print('SUCCESS: all cases agree. Summary: ' + str(args.output / 'summary.json'), flush=True)


if __name__ == '__main__':
    main()
