#!/usr/bin/env python3
"""Profile a resident production MPS model: ordinary timings, then GPU stage probes.

Uses initialized Asimov data (-a), without minimization/HESSE. Overrides and input
selection are preserved. GPU probes synchronize stages and are reported separately.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
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
SECONDS_FIELDS = (
    'parameterUploadSeconds', 'externalWeightUploadSeconds', 'commandEncodeSeconds',
    'deviceWaitSeconds', 'devicePropagationSeconds', 'histogramReadbackSeconds',
    'likelihoodHostSeconds', 'eventWeightReadbackSeconds', 'eventStateReadbackSeconds',
    'cachedDialStageSeconds', 'eventWeightsStageSeconds', 'histogramStageSeconds')
COUNT_FIELDS = (
    'externalWeightUploadBytes', 'externalWeightUploadBlocks', 'histogramReadbackBytes',
    'eventWeightReadbackBytes', 'eventStateReadbackBytes', 'rebinnedEventCount',
    'binCandidateCount', 'binContributionCount', 'contributionBufferBytes')


def stats(values):
    ordered = sorted(values)
    return {'medianMs': statistics.median(values) * 1000,
            'meanMs': statistics.mean(values) * 1000,
            'p95Ms': ordered[int(.95 * (len(ordered) - 1))] * 1000}


def summarize(rows):
    names = sorted(set().union(*(r['seconds'] for r in rows)))
    return {name: stats([r['seconds'][name] for r in rows if name in r['seconds']])
            for name in names}


def histogram(engine):
    engine.getBackendsManager().materialize('Histograms')
    return [(v.sumWeights, v.sqrtSumSqWeights)
            for sample in engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
            for v in sample.getHistogram().getBinContentList()]


def load_engine(args):
    import GUNDAM
    if not hasattr(GUNDAM.BackendManager, 'getDiagnosticMpsStageGpuSeconds'):
        raise RuntimeError('Rebuild GUNDAM with the Python interface and MPS diagnostics; check PYTHONPATH.')
    missing = [name for name in (*SECONDS_FIELDS, *COUNT_FIELDS)
               if not hasattr(GUNDAM.BackendTimingSummary, name)]
    if missing:
        raise RuntimeError(f'Rebuild/select the matching GUNDAM module: missing timing fields {missing}')
    progress(f'Python module: {GUNDAM.__file__}')
    GUNDAM.setRuntimeWorkingDirectory(str(args.work_dir))
    GUNDAM.setNumberOfThreads(args.threads)
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setIsDebug(False)
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfig(str(args.config))
    if args.override:
        builder.override(args.override)
    config = json.loads(builder.getConfig().toString())['fitterEngineConfig']
    config.update(enablePca=False, scaleParStepWithChi2Response=False,
                  generateSamplePlots=False, generateOneSigmaPlots=False)
    backend = config['likelihoodInterfaceConfig'].setdefault('backendManagerConfig', {})
    backend.update(isEnabled=True, type='MPS')
    if args.propagation_mode:
        backend.setdefault('backendConfig', {})['propagationMode'] = args.propagation_mode
    write_json(args.output / 'resolved-fitter-config.json', config)
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    start = time.monotonic()
    progress('Loading dataset once; input selections/limits come from your configuration')
    engine.initialize()
    manager = engine.getBackendsManager()
    manager.setEnableAutoMaterialize(False)
    if not manager.hasBackend() or manager.getType() != 'MPS':
        raise RuntimeError('Expected an active MPS backend')
    manager.setDiagnosticMpsStageProfiling(False)
    engine.evaluateLikelihood()
    metadata = json.loads(manager.getDiagnosticMps())
    write_json(args.output / 'model.json', metadata)
    if not metadata['available'] or metadata['path'] == 'hostFallback':
        raise RuntimeError(f'MPS GPU propagation unavailable: {metadata}')
    progress(f"Initialized in {time.monotonic() - start:.1f} s: {metadata['device']}, "
             f"{metadata['events']} events, {metadata['uniqueDials']} unique dials, path={metadata['path']}")
    return engine


def evaluate_pass(args, engine, sequence, scenario, calls, seed, profiling=False, spread=False):
    manager = engine.getBackendsManager()
    manager.setDiagnosticMpsStageProfiling(False)
    sequence.reset()
    engine.evaluateLikelihood()  # Same initialized state/cache before each pass.
    manager.setDiagnosticMpsStageProfiling(profiling)
    rows, rng, digest = [], random.Random(seed), hashlib.sha256()
    last_progress = time.monotonic()
    try:
        for call in range(calls):
            # Short probes sample coordinates across the whole fit list.
            logical_call = call * max(1, len(sequence.entries) // calls) if spread else call
            start = time.perf_counter()
            changed = sequence.apply(scenario, rng, logical_call, digest)
            update_seconds = time.perf_counter() - start
            start = time.perf_counter()
            engine.evaluateLikelihood()
            elapsed = time.perf_counter() - start
            value = engine.getLikelihoodInterface().getLastLikelihood()
            if not math.isfinite(value):
                raise RuntimeError(f'{scenario}: nonfinite likelihood at evaluation {call}')
            timing = manager.getLastTimingSummary()
            seconds = {name: getattr(timing, name) for name in SECONDS_FIELDS}
            seconds.update(evaluationSeconds=elapsed, parameterUpdateSeconds=update_seconds,
                           hostPreparationSeconds=manager.getLastPreparationSeconds())
            if profiling:
                seconds.update({'gpu:' + name: value for name, value in
                                json.loads(manager.getDiagnosticMpsStageGpuSeconds()).items()})
            if timing.eventWeightReadbackBytes or timing.eventStateReadbackBytes:
                raise RuntimeError('Unexpected event readback during resident evaluations')
            rows.append({'seconds': seconds, 'counts': {name: getattr(timing, name) for name in COUNT_FIELDS},
                         'likelihood': value, 'changedFitCoordinates': changed,
                         'sequenceDigest': digest.hexdigest()})
            if time.monotonic() - last_progress >= 5:
                progress(f'{scenario}: {call + 1}/{calls} evaluations (stage probes={profiling})')
                last_progress = time.monotonic()
    finally:
        manager.setDiagnosticMpsStageProfiling(False)
    return {'rows': rows, 'timings': summarize(rows), 'histogram': histogram(engine)}


def check_equal(reference, actual):
    if len(reference['rows']) != len(actual['rows']):
        raise RuntimeError('Probe/reference evaluation counts differ')
    for left, right in zip(reference['rows'], actual['rows']):
        if left['sequenceDigest'] != right['sequenceDigest'] or not math.isclose(
                left['likelihood'], right['likelihood'], rel_tol=2e-5, abs_tol=2e-5):
            raise RuntimeError('Stage profiling changed the parameter sequence or likelihood')
    if len(reference['histogram']) != len(actual['histogram']) or any(
            not math.isclose(a, b, rel_tol=2e-5, abs_tol=2e-5)
            for left, right in zip(reference['histogram'], actual['histogram'])
            for a, b in zip(left, right)):
        raise RuntimeError('Stage profiling changed histogram contents')


def measure(args):
    import GUNDAM
    engine = load_engine(args)
    sequence = scheduling.ParameterSequence(engine, args.step_scale)
    write_json(args.output / 'parameters.json', sequence.manifest())
    summary = {'complete': False, 'scenarios': {}, 'pythonModule': GUNDAM.__file__,
               'notes': ['GPU stages use separate synchronized commands only in the probe pass.',
                         'Timers overlap: GPU time and CPU wait must not be added.',
                         'No hardware bandwidth/cache-miss/occupancy counters are collected.']}
    write_json(args.output / 'summary.json', summary)
    for scenario in args.scenarios:
        calls = scheduling.scenario_calls(args, scenario, len(sequence.entries))
        progress(f'{scenario}: {calls} calls/trial, {args.trials} trials; warming up')
        if args.warmup:
            evaluate_pass(args, engine, sequence, scenario, args.warmup, args.seed + 1000000)
        trials = []
        for trial in range(args.trials):
            progress(f'{scenario}: ordinary trial {trial + 1}/{args.trials}')
            result = evaluate_pass(args, engine, sequence, scenario, calls, args.seed + trial)
            write_json(args.output / f'{scenario}-trial{trial + 1}.json', result)
            trials.append(result)
            summary['scenarios'][scenario] = {
                'callsPerTrial': calls, 'completedTrials': len(trials),
                'timings': summarize([row for t in trials for row in t['rows']])}
            write_json(args.output / 'summary.json', summary)
        if args.probe_calls:
            progress(f'{scenario}: separate reference + {args.probe_calls} synchronized GPU probes')
            reference = evaluate_pass(args, engine, sequence, scenario, args.probe_calls, args.seed,
                                      spread=True)
            probes = evaluate_pass(args, engine, sequence, scenario, args.probe_calls, args.seed,
                                   profiling=True, spread=True)
            write_json(args.output / f'{scenario}-probes.json', probes)
            check_equal(reference, probes)
            stages = {name: values for name, values in probes['timings'].items() if name.startswith('gpu:')}
            if not stages:
                raise RuntimeError('No GPU stage timestamps returned; check the loaded GUNDAM build')
            summary['scenarios'][scenario].update(probeTimings=stages, probeMatchesReference=True)
        entry = summary['scenarios'][scenario]
        progress(f"{scenario}: evaluation {entry['timings']['evaluationSeconds']['medianMs']:.3f} ms; "
                 f"GPU {entry['timings']['devicePropagationSeconds']['medianMs']:.3f} ms")
        write_json(args.output / 'summary.json', summary)
    summary['complete'] = True
    write_json(args.output / 'summary.json', summary)
    progress(f'SUCCESS: results in {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--override', nargs='*', default=[])
    parser.add_argument('--work-dir', type=Path, default=Path.cwd())
    parser.add_argument('--propagation-mode', choices=['cachedResponses', 'recomputeResponses'],
                        help='Override the MPS response strategy (default comes from the configuration)')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--calls', type=int, default=30)
    parser.add_argument('--global-calls', type=int, default=10)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--probe-calls', type=int, default=8)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--step-scale', type=float, default=.1)
    parser.add_argument('--scenarios', nargs='+', choices=['constant', 'local1', 'local2', 'global'],
                        default=['constant', 'local1', 'local2', 'global'])
    parser.add_argument('--output', type=Path,
                        default=Path(time.strftime('mps-profiling-%Y%m%d-%H%M%S-') + str(os.getpid())))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != 'darwin':
        parser.error('MPS profiling requires macOS/Metal')
    if min(args.threads, args.calls, args.global_calls, args.trials) <= 0 or min(args.warmup, args.probe_calls) < 0:
        parser.error('Counts must be positive (warmup/probe-calls may be zero)')
    if not math.isfinite(args.step_scale) or args.step_scale <= 0:
        parser.error('--step-scale must be finite and positive')
    args.scenarios = list(dict.fromkeys(args.scenarios))
    args.work_dir = args.work_dir.resolve()
    args.config = (args.work_dir / args.config).resolve()
    args.output = args.output.resolve()
    if args.worker:
        measure(args)
        return
    if not args.config.is_file():
        parser.error(f'Configuration does not exist: {args.config}')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'settings.json', {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'host': platform.node(), 'platform': platform.platform(), 'python': sys.executable})
    log_path = args.output / 'native.log'
    progress(f'Results: {args.output}')
    with log_path.open('w') as log:
        command = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--worker',
                   '--output', str(args.output), '--work-dir', str(args.work_dir)]
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        def forward():
            for line in child.stdout:
                log.write(line)
                log.flush()
                if '[benchmark]' in line:
                    print(line.rstrip(), flush=True)
        reader = threading.Thread(target=forward)
        reader.start()
        try:
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    progress(f'Worker PID {child.pid} running; detailed output: {log_path}')
        except KeyboardInterrupt:
            child.terminate()
            child.wait()
            raise
        finally:
            reader.join()
        if code:
            raise RuntimeError(f'Worker exited with code {code}; see {log_path}')


if __name__ == '__main__':
    main()
