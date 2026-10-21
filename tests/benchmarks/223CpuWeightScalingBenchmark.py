#!/usr/bin/env python3
"""Measure resident CPU event-weight scaling with one model load.

Only the number of active weight workers changes. The resident pool, histogram
workers, allocation and CPU affinity stay fixed. Worker probes run separately
from the uninstrumented timing trials. No minimization or HESSE is run.
"""
import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    'scheduling', Path(__file__).with_name('222CpuWeightSchedulingBenchmark.py'))
scheduling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduling)
progress, write_json = scheduling.progress, scheduling.write_json


def stats(values):
    ordered = sorted(values)
    return {'medianMs': statistics.median(values) * 1000,
            'meanMs': statistics.mean(values) * 1000,
            'p95Ms': ordered[int(.95 * (len(ordered) - 1))] * 1000}


def measure(args):
    import GUNDAM
    if not hasattr(GUNDAM.BackendManager, 'setDiagnosticCpuWeightWorkers'):
        raise RuntimeError('Rebuild/install GUNDAM: CPU weight worker diagnostics are missing.')
    engine = scheduling.load_engine(args, args.threads)
    manager = engine.getBackendsManager()
    metadata = json.loads(manager.setDiagnosticCpuWeightScheduling('Dynamic', 2))
    engine.evaluateLikelihood()  # Prime dial and binning caches before changing worker counts.
    reference_llh = engine.getLikelihoodInterface().getLastLikelihood()
    manager.materialize('Histograms')

    def histogram():
        return [(v.sumWeights, v.sqrtSumSqWeights)
                for sample in engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
                for v in sample.getHistogram().getBinContentList()]

    reference_histogram = histogram()
    system = {'host': platform.node(), 'platform': platform.platform(), 'pid': os.getpid(),
              'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
              'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')},
              'model': metadata, 'autoMaterializeDuringTiming': False,
              'policy': 'constant initialized point; resident pool fixed; only weight workers change'}
    if sys.platform.startswith('linux'):
        for label, command in [('cpuTopology', ['lscpu', '-J', '-e=CPU,CORE,SOCKET,NODE']),
                               ('numaHardware', ['numactl', '--hardware'])]:
            try:
                captured = subprocess.run(command, capture_output=True, text=True, timeout=5)
                system[label] = {'exitCode': captured.returncode, 'stdout': captured.stdout[:65536],
                                 'stderr': captured.stderr[:4096]}
            except (OSError, subprocess.TimeoutExpired) as error:
                system[label] = {'unavailable': str(error)}
    write_json(args.output / 'model.json', system)
    results = []
    for round_index in range(args.rounds):
        order = args.workers if round_index % 2 == 0 else list(reversed(args.workers))
        for workers in order:
            tag = f'round{round_index + 1}-w{workers}'
            progress(f'{tag}: warmup; dataset and response cache retained')
            layout = json.loads(manager.setDiagnosticCpuWeightWorkers(workers, False))
            for _ in range(args.warmup):
                engine.evaluateLikelihood()
            raw = []
            for trial in range(args.trials):
                times = {k: [] for k in ('weights', 'cache', 'histograms', 'evaluation')}
                for _ in range(args.calls):
                    start = time.perf_counter()
                    engine.evaluateLikelihood()
                    times['evaluation'].append(time.perf_counter() - start)
                    timing = manager.getLastTimingSummary()
                    times['weights'].append(timing.eventWeightsStageSeconds)
                    times['cache'].append(timing.cachedDialStageSeconds)
                    times['histograms'].append(timing.histogramStageSeconds)
                    if timing.cacheCandidateCount:
                        raise RuntimeError('Constant-point cache is not warm: unexpected dial candidates.')
                    value = engine.getLikelihoodInterface().getLastLikelihood()
                    if not math.isclose(value, reference_llh, rel_tol=1e-10, abs_tol=1e-10):
                        raise RuntimeError('Likelihood changed with weight worker count.')
                raw.append(times)
                write_json(args.output / f'{tag}.partial.json', {'complete': False, 'raw': raw})
                progress(f"{tag}: trial {trial + 1}/{args.trials}, weights {stats(times['weights'])['medianMs']:.3f} ms")
            # Enable per-worker clocks only for this separate diagnostic pass.
            manager.setDiagnosticCpuWeightWorkers(workers, True)
            probes = []
            for _ in range(args.probe_calls):
                engine.evaluateLikelihood()
                diagnostic = json.loads(manager.getDiagnosticCpuWeightScheduling())
                records = diagnostic['lastWeightWorkers']
                if sum(r['events'] for r in records) != metadata['events'] or sum(
                        r['dialReferences'] for r in records) != metadata['eventDialReferences']:
                    raise RuntimeError('Worker probes do not cover the complete model.')
                maximum = max(r['wallSeconds'] for r in records)
                probes.append({'workers': records,
                               'stageSeconds': manager.getLastTimingSummary().eventWeightsStageSeconds,
                               'meanToMaxWorkerTime': statistics.mean(r['wallSeconds'] for r in records) / maximum
                               if maximum > 0 else None})
            manager.setDiagnosticCpuWeightWorkers(workers, False)
            manager.materialize('Histograms')
            if histogram() != reference_histogram:
                raise RuntimeError('Histogram sums or squared sums changed with weight worker count.')
            flattened = {k: [v for trial in raw for v in trial[k]] for k in raw[0]}
            result = {'complete': True, 'round': round_index + 1, 'workers': workers,
                      'poolThreads': args.threads, 'layout': layout, 'raw': raw, 'probes': probes,
                      'timings': {k: stats(v) for k, v in flattened.items()},
                      'likelihoodsAgree': True, 'histogramsExactlyAgree': True}
            write_json(args.output / f'{tag}.json', result)
            (args.output / f'{tag}.partial.json').unlink(missing_ok=True)
            compact = {k: v for k, v in result.items() if k not in ('raw', 'probes', 'layout')}
            compact['referencesPerSecond'] = metadata['eventDialReferences'] / statistics.median(flattened['weights'])
            results.append(compact)
            for r in results:
                baseline = next((b for b in results if b['round'] == r['round'] and b['workers'] == 1), None)
                if baseline:
                    r['weightSpeedupVsOneWorker'] = baseline['timings']['weights']['medianMs'] / r['timings']['weights']['medianMs']
            write_json(args.output / 'summary.json', results)
    progress(f'SUCCESS: all worker counts preserve LLH and histograms. Results: {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--override', nargs='*', default=[])
    parser.add_argument('--work-dir', type=Path, default=Path.cwd())
    parser.add_argument('--threads', type=int, default=8, help='Resident pool size; dataset is loaded once')
    parser.add_argument('--workers', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--calls', type=int, default=30)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=2, help='Alternate ascending and descending order')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--probe-calls', type=int, default=5)
    parser.add_argument('--output', type=Path, default=Path(time.strftime('cpu-weight-scaling-%Y%m%d-%H%M%S-') + str(os.getpid())))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.threads, *args.workers, args.calls, args.trials, args.rounds, args.warmup, args.probe_calls) < 1:
        parser.error('Counts must be positive')
    if max(args.workers) > args.threads:
        parser.error('--workers cannot exceed the resident --threads pool')
    args.workers = sorted(set(args.workers))
    args.work_dir = args.work_dir.resolve()
    args.config = (args.work_dir / args.config).resolve()
    args.output = args.output.resolve()
    if args.worker:
        measure(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    settings = vars(args).copy()
    write_json(args.output / 'settings.json', {k: str(v) if isinstance(v, Path) else v for k, v in settings.items()})
    progress(f'Results: {args.output}; one dataset load, no event-count limit added')
    count = len(args.workers) * args.rounds
    progress(f'{count * args.calls * args.trials} timed evaluations + {count * args.probe_calls} separate worker probes + warmup/priming')
    log_path = args.output / 'native.log'
    command = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--worker', '--output', str(args.output)]
    environment = os.environ.copy()
    environment.pop('GUNDAM_CPU_DIAGNOSTIC_BLOCKS_PER_THREAD', None)
    with log_path.open('w') as log, subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                    text=True, env=environment) as child:
        def forward():
            for line in child.stdout:
                log.write(line)
                log.flush()
                if '[benchmark]' in line:
                    print(line.rstrip(), flush=True)
        reader = threading.Thread(target=forward)
        reader.start()
        while True:
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                progress(f'PID {child.pid} still running; native log: {log_path}')
        reader.join()
        if code:
            raise RuntimeError(f'Benchmark worker exited {code}; see {log_path}')


if __name__ == '__main__':
    main()
