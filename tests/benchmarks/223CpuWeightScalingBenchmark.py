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


def summarize_hardware(probes):
    reports = [worker['weightProbe'] for probe in probes for worker in probe['workers']]
    names = ('cycles', 'instructions', 'cacheReferences', 'cacheMisses', 'llcReadMisses', 'dtlbReadMisses')
    out = {'workerSamples': len(reports), 'validSamplesByEvent': {
        name: sum(r['events'][name]['scaledCount'] is not None for r in reports) for name in names}}
    fractions = [e['runningFraction'] for r in reports for e in r['events'].values()
                 if e['runningFraction'] is not None]
    out['minimumRunningFraction'] = min(fractions) if fractions else None
    cpu = [r['cpuSeconds'] / r['wallSeconds'] for r in reports
           if r.get('cpuSeconds') is not None and r.get('wallSeconds', 0) > 0]
    out['medianWorkerCpuToWall'] = statistics.median(cpu) if cpu else None
    ipc, misses, instructions = [], [], []
    for probe in probes:
        workers = probe['workers']
        def total(name):
            values = [w['weightProbe']['events'][name]['scaledCount'] for w in workers]
            return sum(values) if all(v is not None for v in values) else None
        cycles, retired, cache_misses = total('cycles'), total('instructions'), total('cacheMisses')
        references = sum(w['dialReferences'] for w in workers)
        if cycles and retired is not None:
            ipc.append(retired / cycles)
        if references and retired is not None:
            instructions.append(retired / references)
        if references and cache_misses is not None:
            misses.append(1000 * cache_misses / references)
    out['medianIpcEstimate'] = statistics.median(ipc) if ipc else None
    out['medianInstructionsPerDialReference'] = statistics.median(instructions) if instructions else None
    out['medianGenericCacheMissesPer1000DialReferences'] = statistics.median(misses) if misses else None
    return out


def measure(args):
    import GUNDAM
    if not hasattr(GUNDAM.BackendTimingSummary, 'likelihoodHostSeconds'):
        raise RuntimeError('Rebuild/install GUNDAM: full stage timing fields are missing.')
    if not hasattr(GUNDAM.BackendManager, 'setDiagnosticCpuWeightWorkers'):
        raise RuntimeError('Rebuild/install GUNDAM: CPU weight worker diagnostics are missing.')
    if args.hardware_counters and (not hasattr(GUNDAM.BackendManager, 'setDiagnosticCpuWeightCounters')
                                      or not hasattr(GUNDAM, 'probeCpuWeightCounters')):
        raise RuntimeError('Rebuild/install GUNDAM: phase-scoped hardware probes are missing.')
    if args.hardware_counters:
        preflight = json.loads(GUNDAM.probeCpuWeightCounters())
        write_json(args.output / 'counter-preflight.json', preflight)
        opened = [name for name, event in preflight['events'].items() if event['available']]
        errors = {name: event.get('error', 'unsupported platform') for name, event in preflight['events'].items()
                  if not event['available']}
        progress(f'Counter preflight before loading: opened {len(opened)}/6 events; {errors}')
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
    if args.hardware_counters:
        path = Path('/proc/sys/kernel/perf_event_paranoid')
        try:
            system['perfEventParanoid'] = path.read_text().strip()
        except OSError:
            system['perfEventParanoid'] = None
        system['hardwareProbePolicy'] = 'calling worker, userspace only, independent perf events; weights only'
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
                times = {k: [] for k in ('weights', 'cache', 'histograms', 'likelihood', 'variables',
                                         'binning', 'hostPreparation', 'evaluation')}
                for _ in range(args.calls):
                    start = time.perf_counter()
                    engine.evaluateLikelihood()
                    times['evaluation'].append(time.perf_counter() - start)
                    timing = manager.getLastTimingSummary()
                    times['weights'].append(timing.eventWeightsStageSeconds)
                    times['cache'].append(timing.cachedDialStageSeconds)
                    times['histograms'].append(timing.histogramStageSeconds)
                    times['likelihood'].append(timing.likelihoodHostSeconds)
                    times['variables'].append(timing.variableStageSeconds)
                    times['binning'].append(timing.binningStageSeconds)
                    times['hostPreparation'].append(manager.getLastPreparationSeconds())
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
            if args.hardware_counters:
                manager.setDiagnosticCpuWeightCounters(True)
                # Open counters on their owner workers before recorded samples.
                engine.evaluateLikelihood()
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
            if args.hardware_counters:
                hardware = summarize_hardware(probes)
                progress(f"{tag}: hardware samples: {hardware['validSamplesByEvent']}; "
                         f"minimum PMU running fraction: {hardware['minimumRunningFraction']}")
                manager.setDiagnosticCpuWeightCounters(False)
            manager.setDiagnosticCpuWeightWorkers(workers, False)
            manager.materialize('Histograms')
            if histogram() != reference_histogram:
                raise RuntimeError('Histogram sums or squared sums changed with weight worker count.')
            flattened = {k: [v for trial in raw for v in trial[k]] for k in raw[0]}
            result = {'complete': True, 'round': round_index + 1, 'workers': workers,
                      'poolThreads': args.threads, 'layout': layout, 'raw': raw, 'probes': probes,
                      'timings': {k: stats(v) for k, v in flattened.items()},
                      'likelihoodsAgree': True, 'histogramsExactlyAgree': True}
            if args.hardware_counters:
                result['hardware'] = hardware
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
    if args.perf_record:
        helper_spec = importlib.util.spec_from_file_location(
            'sampling', Path(__file__).with_name('CpuSamplingProfile.py'))
        sampling = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(sampling)
        def evaluate_checked():
            engine.evaluateLikelihood()
            if not math.isclose(engine.getLikelihoodInterface().getLastLikelihood(), reference_llh,
                                rel_tol=1e-10, abs_tol=1e-10):
                raise RuntimeError('Likelihood changed during sampling pass.')
        sampling.record_profile(args.output, evaluate_checked, args.profile_calls, write_json, progress)
        manager.materialize('Histograms')
        if histogram() != reference_histogram:
            raise RuntimeError('Histogram changed during sampling pass.')
    progress(f'SUCCESS: all worker counts preserve LLH and histograms. Results: {args.output}')


def main(single_thread=False, description=None):
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--override', nargs='*', default=[])
    parser.add_argument('--work-dir', type=Path, default=Path.cwd())
    parser.add_argument('--threads', type=int, default=1 if single_thread else 8, help='Resident pool size; dataset is loaded once')
    parser.add_argument('--workers', nargs='+', type=int, default=[1] if single_thread else [1, 2, 4, 8])
    parser.add_argument('--calls', type=int, default=40 if single_thread else 30)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=1 if single_thread else 2, help='Alternate ascending and descending order')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--probe-calls', type=int, default=5)
    parser.add_argument('--hardware-counters', action='store_true',
                        help='Optional Linux perf counters and thread CPU/context-switch probes, only in the separate probe pass')
    if single_thread:
        parser.add_argument('--perf-record', action='store_true', help='Optional Linux perf function/call-stack sampling, after timing trials')
        parser.add_argument('--profile-calls', type=int, default=40, help='Separate sampled evaluations')
    else:
        parser.set_defaults(perf_record=False, profile_calls=40)
    prefix = 'cpu-single-thread-' if single_thread else 'cpu-weight-scaling-'
    parser.add_argument('--output', type=Path, default=Path(time.strftime(prefix + '%Y%m%d-%H%M%S-') + str(os.getpid())))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if single_thread and (args.threads != 1 or args.workers != [1]):
        parser.error('This benchmark requires --threads 1 --workers 1')
    if min(args.threads, *args.workers, args.calls, args.trials, args.rounds, args.warmup, args.probe_calls, args.profile_calls) < 1:
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
    if args.perf_record:
        progress(f'Additional separate perf pass: {args.profile_calls} evaluations if perf can attach')
    log_path = args.output / 'native.log'
    command = [sys.executable, '-u', str(Path(sys.argv[0]).resolve()), *sys.argv[1:], '--worker', '--output', str(args.output)]
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
