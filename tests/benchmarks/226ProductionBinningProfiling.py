#!/usr/bin/env python3
"""Compare CPU/MPS production rebinning, preserving the configured external worker.

Load Asimov inputs once per backend/smoothing case; vary only a named parameter
set. Ordinary timings and synchronized GPU stage probes are separate passes.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import platform
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    'profiling', Path(__file__).with_name('225MpsBackendProfiling.py'))
profiling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiling)
progress, write_json = profiling.progress, profiling.write_json
SECONDS = (*profiling.SECONDS_FIELDS, 'variableStageSeconds', 'binningStageSeconds')


def load_engine(args, output):
    import GUNDAM
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
    li = config['likelihoodInterfaceConfig']
    li.setdefault('backendManagerConfig', {}).update(isEnabled=True, type=args.backend)
    samples = li['propagatorConfig']['sampleSetConfig']['sampleList']
    if args.mode == 'hard':
        for sample in samples:
            sample.pop('binningResponse', None)
    elif not any(sample.get('binningResponse', {}).get('isEnabled', True)
                 for sample in samples if 'binningResponse' in sample):
        raise RuntimeError('Smoothed case requires binningResponse in the input overrides')
    write_json(output / 'resolved-fitter-config.json', config)
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    progress(f'{args.backend}-{args.mode}: loading Asimov inputs with {GUNDAM.__file__}')
    start = time.monotonic()
    engine.initialize()
    manager = engine.getBackendsManager()
    manager.setEnableAutoMaterialize(False)
    if not manager.hasBackend() or manager.getType() != args.backend:
        raise RuntimeError('Requested backend was not activated')
    engine.evaluateLikelihood()
    if args.backend == 'MPS':
        model = json.loads(manager.getDiagnosticMps())
        write_json(output / 'model.json', model)
        if not model['available'] or model['path'] != 'eventPropagation':
            raise RuntimeError(f'Expected GPU event propagation: {model.get("path")}')
    progress(f'{args.backend}-{args.mode}: initialized in {time.monotonic() - start:.1f} s')
    return engine


def histogram(engine):
    values = profiling.histogram(engine)
    samples = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
    # Include the smoothing mass outside the histogram support as well.
    return values + [(s.getHistogram().getOutsideBinContent().sumWeights,
                      s.getHistogram().getOutsideBinContent().sqrtSumSqWeights) for s in samples]


def bin_movement(engine, sequence, scenario, calls, seed):
    manager = engine.getBackendsManager()
    def bins():
        manager.materialize('EventState')
        return [event.getIndices().bin
                for sample in engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
                for event in sample.getEventList()]
    sequence.reset()
    engine.evaluateLikelihood()
    previous = bins()
    rows, rng, digest = [], random.Random(seed), hashlib.sha256()
    for call in range(calls):
        sequence.apply(scenario, rng, call, digest)
        engine.evaluateLikelihood()
        dirty = manager.getLastTimingSummary().rebinnedEventCount
        current = bins()
        changed = sum(a != b for a, b in zip(previous, current))
        rows.append({'rebinnedEvents': dirty, 'changedHardBins': changed,
                     'sameBinFractionAmongRebinned': (dirty - changed) / dirty if dirty else None})
        previous = current
    return rows


def measure(engine, sequence, scenario, calls, seed, probe=False, capture=False):
    manager = engine.getBackendsManager()
    sequence.reset()
    engine.evaluateLikelihood()
    if probe:
        manager.setDiagnosticMpsStageProfiling(True)
    rows, histograms, rng, digest = [], [], random.Random(seed), hashlib.sha256()
    last_progress = time.monotonic()
    try:
        for call in range(calls):
            sequence.apply(scenario, rng, call, digest)
            start = time.perf_counter()
            engine.evaluateLikelihood()
            elapsed = time.perf_counter() - start
            timing = manager.getLastTimingSummary()
            value = engine.getLikelihoodInterface().getLastLikelihood()
            if not math.isfinite(value):
                raise RuntimeError('Nonfinite likelihood')
            seconds = {name: getattr(timing, name) for name in SECONDS}
            seconds.update(evaluationSeconds=elapsed,
                           hostPreparationSeconds=manager.getLastPreparationSeconds())
            if probe:
                seconds.update({'gpu:' + name: value for name, value in
                                json.loads(manager.getDiagnosticMpsStageGpuSeconds()).items()})
            if timing.eventWeightReadbackBytes or timing.eventStateReadbackBytes:
                raise RuntimeError('Unexpected event readback during measurement')
            rows.append({'seconds': seconds, 'likelihood': value,
                         'sequenceDigest': digest.hexdigest(),
                         'counts': {name: getattr(timing, name) for name in profiling.COUNT_FIELDS}})
            # Correctness snapshots happen after the timer. Only separate
            # ordinary trials without snapshots contribute to timing summaries.
            if capture:
                histograms.append(histogram(engine))
            if call + 1 == calls or time.monotonic() - last_progress >= 5:
                progress(f'{scenario}: {call + 1}/{calls}, {elapsed * 1000:.2f} ms (probe={probe})')
                last_progress = time.monotonic()
    finally:
        if probe:
            manager.setDiagnosticMpsStageProfiling(False)
    return {'rows': rows, 'timings': profiling.summarize(rows),
            'histogram': histogram(engine), 'histograms': histograms}


def worker(args):
    output = args.output / f'{args.backend}-{args.mode}'
    engine = load_engine(args, output)
    sequence = profiling.scheduling.ParameterSequence(engine, args.step_scale)
    sets = engine.getLikelihoodInterface().getModelPropagator().getParametersManager().getParameterSetsList()
    titles = {p.getFullTitle() for parameter_set in sets
              if parameter_set.getName() == args.parameter_set
              for p in parameter_set.getEffectiveParameterList()}
    sequence.entries = [entry for entry in sequence.entries if entry[0].getFullTitle() in titles]
    if not sequence.entries:
        raise RuntimeError(f'No variable effective parameters in {args.parameter_set!r}')
    sequence.reset()
    write_json(output / 'parameters.json', sequence.manifest())
    summary = {'complete': False, 'scenarios': {}}
    for scenario in args.scenarios:
        if args.warmup:
            measure(engine, sequence, scenario, args.warmup, args.seed + 1000000)
        rows = []
        for trial in range(args.trials):
            progress(f'{args.backend}-{args.mode}: {scenario} trial {trial + 1}/{args.trials}')
            result = measure(engine, sequence, scenario, args.calls, args.seed + trial)
            write_json(output / f'{scenario}-trial{trial + 1}.json', result)
            rows.extend(result['rows'])
            summary['scenarios'][scenario] = {'timings': profiling.summarize(rows),
                'meanCounts': {name: sum(row['counts'][name] for row in rows) / len(rows)
                               for name in profiling.COUNT_FIELDS}}
            write_json(output / 'summary.json', summary)
        # Replay a short reference for every backend and retain every histogram.
        reference = measure(engine, sequence, scenario, args.probe_calls, args.seed, capture=True)
        write_json(output / f'{scenario}-reference.json', reference)
        if args.backend == 'MPS':
            probe = measure(engine, sequence, scenario, args.probe_calls, args.seed, probe=True, capture=True)
            profiling.check_equal(reference, probe)
            write_json(output / f'{scenario}-probes.json', probe)
    if args.mode == 'hard' and args.backend == 'MPS':
        for scenario in args.scenarios:
            write_json(output / f'{scenario}-bin-movement.json',
                       bin_movement(engine, sequence, scenario, args.probe_calls, args.seed))
    summary['complete'] = True
    write_json(output / 'summary.json', summary)


def supervise(args, backend, mode):
    output = args.output / f'{backend}-{mode}'
    output.mkdir()
    command = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:],
               '--worker', '--backend', backend, '--mode', mode,
               '--output', str(args.output), '--work-dir', str(args.work_dir)]
    with (output / 'native.log').open('w') as log:
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
                    progress(f'{backend}-{mode}: running PID {child.pid}; details in {output / "native.log"}')
        except KeyboardInterrupt:
            child.terminate()
            child.wait()
            raise
        finally:
            reader.join()
    if code:
        raise RuntimeError(f'{backend}-{mode} failed ({code}); see {output / "native.log"}')


def compare(args):
    checks, cases = {}, {}
    for mode in args.modes:
        for backend in args.backends:
            path = args.output / f'{backend}-{mode}'
            case = json.loads((path / 'summary.json').read_text())
            if not case['complete']:
                raise RuntimeError(f'Incomplete case: {path}')
            for scenario in args.scenarios:
                rows = [row for file in path.glob(f'{scenario}-trial*.json')
                        for row in json.loads(file.read_text())['rows']]
                case['scenarios'][scenario]['timings']['evaluationMinusPreparationSeconds'] = profiling.stats([
                    row['seconds']['evaluationSeconds'] - row['seconds']['hostPreparationSeconds']
                    for row in rows])
            cases[f'{backend}-{mode}'] = case
    for mode in args.modes:
        if not {'CPU', 'MPS'}.issubset(args.backends):
            continue
        for scenario in args.scenarios:
            paths = [args.output / f'{backend}-{mode}' / f'{scenario}-reference.json'
                     for backend in ('CPU', 'MPS')]
            manifests = [json.loads((args.output / f'{backend}-{mode}' / 'parameters.json').read_text())
                         for backend in ('CPU', 'MPS')]
            if manifests[0] != manifests[1]:
                raise RuntimeError('CPU/MPS parameter manifests differ')
            cpu, mps = [json.loads(path.read_text()) for path in paths]
            profiling.check_equal(cpu, mps)
            pairs = [(a, b) for left, right in zip(cpu['histograms'], mps['histograms'])
                     for bin_a, bin_b in zip(left, right) for a, b in zip(bin_a, bin_b)]
            if len(cpu['histograms']) != len(mps['histograms']) or any(
                    len(left) != len(right) for left, right in zip(cpu['histograms'], mps['histograms'])):
                raise RuntimeError('CPU/MPS histogram dimensions differ')
            if any(not math.isclose(a, b, rel_tol=2e-5, abs_tol=2e-5) for a, b in pairs):
                raise RuntimeError('CPU/MPS per-evaluation histograms differ')
            checks[f'{mode}-{scenario}'] = {'passed': True, 'relativeTolerance': 2e-5,
                'absoluteTolerance': 2e-5, 'maxHistogramAbsoluteDifference': max(
                    (abs(a - b) for a, b in pairs), default=0)}
    write_json(args.output / 'cross-backend-checks.json', checks)
    write_json(args.output / 'summary.json', {'complete': True, 'cases': cases, 'checks': checks})
    progress('All requested cases complete; available cross-backend comparisons passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--override', nargs='*', default=[])
    parser.add_argument('--output', type=Path, default=Path(time.strftime('mps-binning-%Y%m%d-%H%M%S')))
    parser.add_argument('--backends', nargs='+', choices=['CPU', 'MPS'], default=['MPS', 'CPU'])
    parser.add_argument('--modes', nargs='+', choices=['hard', 'smoothed'], default=['hard', 'smoothed'])
    parser.add_argument('--parameter-set', default='Cross-Section Systematics Eb')
    parser.add_argument('--scenarios', nargs='+', choices=['constant', 'local1', 'global'],
                        default=['constant', 'local1', 'global'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--calls', type=int, default=10)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--probe-calls', type=int, default=5)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--step-scale', type=float, default=.1)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--backend', choices=['CPU', 'MPS'], help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=['hard', 'smoothed'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.threads, args.calls, args.trials, args.probe_calls) <= 0 or args.warmup < 0:
        parser.error('Counts must be positive; warmup may be zero')
    if not math.isfinite(args.step_scale) or args.step_scale <= 0:
        parser.error('step-scale must be finite and positive')
    if 'MPS' in args.backends and sys.platform != 'darwin':
        parser.error('MPS requires macOS/Metal; select --backends CPU on other platforms')
    args.backends = list(dict.fromkeys(args.backends))
    args.modes = list(dict.fromkeys(args.modes))
    args.scenarios = list(dict.fromkeys(args.scenarios))
    args.work_dir = args.work_dir.resolve()
    args.config = (args.work_dir / args.config).resolve()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'settings.json', {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'python': sys.executable, 'platform': platform.platform()})
    for mode in dict.fromkeys(args.modes):
        for backend in dict.fromkeys(args.backends):
            supervise(args, backend, mode)
    compare(args)


if __name__ == '__main__':
    main()
