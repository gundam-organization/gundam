#!/usr/bin/env python3
"""Profile resident production CUDA likelihood evaluations and compare response modes.

By default cases share one loaded Asimov model. No fit/Hesse is performed.
Use --execution-mode isolated for a fresh process/model per case.
CUDA stages use the backend's existing stream events, without extra stage fences.
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
CASES = ('CUDA-cachedResponses', 'CUDA-recomputeResponses', 'CPU')
SECONDS_FIELDS = ('parameterUploadSeconds', 'cachedDialStageSeconds',
                  'cachePreparationSeconds',
                  'eventWeightsStageSeconds', 'histogramStageSeconds',
                  'devicePropagationSeconds', 'histogramReadbackSeconds',
                  'likelihoodHostSeconds', 'eventWeightReadbackSeconds')
COUNT_FIELDS = ('histogramReadbackBytes', 'eventWeightReadbackBytes', 'eventStateReadbackBytes',
                'cacheCandidateCount')


def stats(values):
    ordered = sorted(values)
    return {'medianMs': statistics.median(values) * 1000,
            'meanMs': statistics.mean(values) * 1000,
            'p95Ms': ordered[int(.95 * (len(ordered) - 1))] * 1000}


def summarize(rows):
    return {name: stats([row['seconds'][name] for row in rows]) for name in rows[0]['seconds']}


def histogram(engine):
    engine.getBackendsManager().materialize('Histograms')
    return [(b.sumWeights, b.sqrtSumSqWeights)
            for sample in engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
            for b in sample.getHistogram().getBinContentList()]


def backend_config(args):
    backend_type = 'CPU' if args.worker == 'CPU' else 'CUDA'
    return {'isEnabled': True, 'type': backend_type,
            'backendConfig': {} if backend_type == 'CPU' else {
                'deviceIndex': args.device_index, 'propagationMode': args.worker.split('-', 1)[1]}}


def load_engine(args):
    import GUNDAM
    progress(f'{args.worker}: Python module {GUNDAM.__file__}')
    missing = [name for name in (*SECONDS_FIELDS, *COUNT_FIELDS)
               if not hasattr(GUNDAM.BackendTimingSummary, name)]
    if missing:
        raise RuntimeError(f'Rebuild/select the matching Python interface: missing {missing}')
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
    # Replace backend-specific overrides so CPU never inherits CUDA-only keys.
    backend = backend_config(args)
    backend_type = backend['type']
    config['likelihoodInterfaceConfig']['backendManagerConfig'] = backend
    write_json(args.output / 'resolved-fitter-config.json', config)
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    progress('Loading dataset once; input selection/limits are preserved')
    start = time.monotonic()
    engine.initialize()
    manager = engine.getBackendsManager()
    manager.setEnableAutoMaterialize(False)
    if not manager.hasBackend() or manager.getType() != backend_type:
        raise RuntimeError(f'Expected an active {backend_type} backend')
    engine.evaluateLikelihood()
    progress(f'{args.worker}: initialized in {time.monotonic() - start:.1f} s')
    return engine


def evaluate_pass(args, engine, sequence, scenario, calls, seed):
    sequence.reset()
    engine.evaluateLikelihood()  # Identical center and primed cache in every trial.
    rng, digest, rows = random.Random(seed), hashlib.sha256(), []
    manager = engine.getBackendsManager()
    last_progress = time.monotonic()
    for call in range(calls):
        indices = list(sequence.coordinates(scenario, call))
        start = time.perf_counter()
        changed = sequence.apply(scenario, rng, call, digest)
        update_seconds = time.perf_counter() - start
        start = time.perf_counter()
        engine.evaluateLikelihood()
        elapsed = time.perf_counter() - start
        value = engine.getLikelihoodInterface().getLastLikelihood()
        timing = manager.getLastTimingSummary()
        seconds = {name: getattr(timing, name) for name in SECONDS_FIELDS}
        seconds.update(evaluationSeconds=elapsed, parameterUpdateSeconds=update_seconds,
                       hostPreparationSeconds=manager.getLastPreparationSeconds())
        if not math.isfinite(value) or any(not math.isfinite(v) or v < 0 for v in seconds.values()):
            raise RuntimeError(f'{scenario}: nonfinite result or invalid timing at call {call}')
        counts = {name: getattr(timing, name) for name in COUNT_FIELDS}
        if counts['eventWeightReadbackBytes'] or counts['eventStateReadbackBytes']:
            raise RuntimeError('Unexpected event readback during resident evaluation')
        rows.append({'seconds': seconds, 'counts': counts, 'likelihood': value,
                     'changedFitCoordinates': changed, 'coordinateIndices': indices,
                     'sequenceDigest': digest.hexdigest()})
        if time.monotonic() - last_progress >= 5:
            progress(f'{args.worker}/{scenario}: {call + 1}/{calls} evaluations')
            last_progress = time.monotonic()
    bins = histogram(engine)  # Explicit materialization is outside all evaluation timers.
    if any(not math.isfinite(v) for pair in bins for v in pair):
        raise RuntimeError('Nonfinite materialized histogram')
    return {'rows': rows, 'timings': summarize(rows), 'histogram': bins}


def measure(args, engine=None, sequence=None):
    import GUNDAM
    if engine is None:
        engine = load_engine(args)
    if sequence is None:
        sequence = scheduling.ParameterSequence(engine, args.step_scale)
    manifest = sequence.manifest()
    write_json(args.output / 'parameters.json', manifest)
    result = {'complete': False, 'case': args.worker, 'pythonModule': GUNDAM.__file__,
              'pythonModuleSha256': hashlib.sha256(Path(GUNDAM.__file__).read_bytes()).hexdigest(),
              'parameters': manifest, 'executionMode': args.execution_mode, 'pid': os.getpid(),
              'scenarios': {}}
    for scenario in args.scenarios:
        calls = scheduling.scenario_calls(args, scenario, len(sequence.entries))
        if args.warmup:
            evaluate_pass(args, engine, sequence, scenario, args.warmup, args.seed + 1000000)
        trials = []
        for trial in range(args.trials):
            progress(f'{args.worker}/{scenario}: trial {trial + 1}/{args.trials}, {calls} calls')
            measurement = evaluate_pass(args, engine, sequence, scenario, calls, args.seed + trial)
            write_json(args.output / f'{scenario}-trial{trial + 1}.json', measurement)
            trials.append(measurement)
        rows = [row for trial in trials for row in trial['rows']]
        timings = summarize(rows)
        # CUDA event durations overlap the synchronous host evaluation. Rank stages
        # within each clock domain; never add GPU time to host wait/evaluation.
        cuda_stages = ('parameterUploadSeconds', 'cachedDialStageSeconds',
                       'eventWeightsStageSeconds', 'histogramStageSeconds')
        gpu_ranking = sorted(cuda_stages, key=lambda key: timings[key]['medianMs'], reverse=True)
        entry = {'callsPerTrial': calls, 'trials': args.trials, 'timings': timings,
                 'medianResponseCandidates': statistics.median(row['counts']['cacheCandidateCount'] for row in rows),
                 'gpuStageRanking': gpu_ranking if args.worker != 'CPU' else [],
                 'coordinateCostsFile': None}
        if scenario == 'local1':
            grouped = [[] for _ in sequence.entries]
            for row in rows:
                grouped[row['coordinateIndices'][0]].append(row)
            costs = [{**manifest[index], 'samples': len(group),
                      'medianResponseCandidates': statistics.median(row['counts']['cacheCandidateCount'] for row in group),
                      'timings': summarize(group)} for index, group in enumerate(grouped)]
            costs.sort(key=lambda row: row['timings']['evaluationSeconds']['medianMs'], reverse=True)
            entry['coordinateCostsFile'] = 'local1-coordinate-costs.json'
            write_json(args.output / entry['coordinateCostsFile'], costs)
        result['scenarios'][scenario] = entry
        write_json(args.output / 'summary.json', result)
        progress(f"{args.worker}/{scenario}: evaluation {timings['evaluationSeconds']['medianMs']:.3f} ms; "
                 f"preparation {timings['hostPreparationSeconds']['medianMs']:.3f} ms; "
                 f"GPU {timings['devicePropagationSeconds']['medianMs']:.3f} ms; "
                 f"response candidates {entry['medianResponseCandidates']}")
    result['complete'] = True
    write_json(args.output / 'summary.json', result)


def measure_resident(args):
    import GUNDAM
    if not hasattr(GUNDAM.BackendManager, 'replaceBackend'):
        raise RuntimeError('Rebuild/reinstall the Python interface: replaceBackend is unavailable')
    root = args.output
    engine = sequence = config = None
    for case in args.cases:
        args.worker = case
        args.output = root / case
        args.output.mkdir()
        start = time.monotonic()
        if engine is None:
            engine = load_engine(args)
            sequence = scheduling.ParameterSequence(engine, args.step_scale)
            config = json.loads((args.output / 'resolved-fitter-config.json').read_text())
        else:
            sequence.reset()
            backend = backend_config(args)
            progress(f'{case}: replacing backend; keeping loaded events, dials and Asimov data')
            engine.getBackendsManager().replaceBackend(
                backend['type'], GUNDAM.JsonType(json.dumps(backend['backendConfig'])))
            config['likelihoodInterfaceConfig']['backendManagerConfig'] = backend
            write_json(args.output / 'resolved-fitter-config.json', config)
            engine.evaluateLikelihood()  # Prime new buffers outside measurements.
        write_json(args.output / 'setup.json', {
            'seconds': time.monotonic() - start, 'includesDatasetLoading': case == args.cases[0],
            'executionMode': 'resident', 'pid': os.getpid()})
        measure(args, engine, sequence)
    args.output = root


def compare(output, cases, scenarios, trials, relative, absolute):
    reference = 'CPU' if 'CPU' in cases else cases[0]
    manifest = json.loads((output / reference / 'parameters.json').read_text())
    comparison = {'reference': reference, 'relativeTolerance': relative,
                  'absoluteTolerance': absolute, 'passed': True, 'cases': {}}
    for case in (case for case in cases if case != reference):
        if manifest != json.loads((output / case / 'parameters.json').read_text()):
            raise RuntimeError(f'{case}: parameter manifests differ')
        entries = {}
        for scenario in scenarios:
            max_llh, max_bin, passed = 0., 0., True
            for trial in range(1, trials + 1):
                name = f'{scenario}-trial{trial}.json'
                left = json.loads((output / reference / name).read_text())
                right = json.loads((output / case / name).read_text())
                if len(left['rows']) != len(right['rows']):
                    raise RuntimeError('Evaluation counts differ')
                for a, b in zip(left['rows'], right['rows']):
                    if a['sequenceDigest'] != b['sequenceDigest']:
                        raise RuntimeError('Parameter sequences differ')
                    max_llh = max(max_llh, abs(a['likelihood'] - b['likelihood']))
                    passed &= math.isclose(a['likelihood'], b['likelihood'], rel_tol=relative, abs_tol=absolute)
                if len(left['histogram']) != len(right['histogram']):
                    raise RuntimeError('Histogram sizes differ')
                for a, b in zip(left['histogram'], right['histogram']):
                    for x, y in zip(a, b):
                        max_bin = max(max_bin, abs(x - y))
                        passed &= math.isclose(x, y, rel_tol=relative, abs_tol=absolute)
            def median(case_name):
                summary = json.loads((output / case_name / 'summary.json').read_text())
                return summary['scenarios'][scenario]['timings']['evaluationSeconds']['medianMs']
            entries[scenario] = {'passed': passed, 'maxAbsoluteLikelihoodDifference': max_llh,
                                 'maxAbsoluteHistogramOrErrorDifference': max_bin,
                                 'speedupOverReference': median(reference) / median(case)}
            comparison['passed'] &= passed
        comparison['cases'][case] = entries
    write_json(output / 'comparison.json', comparison)
    if not comparison['passed']:
        raise RuntimeError('Numerical comparison failed; see comparison.json')


def write_report(output, cases):
    lines = ['# Resident likelihood profiling', '',
             'Execution mode: ' + json.loads((output / 'settings.json').read_text())['execution_mode'] + '.', '',
             'All durations below are medians in milliseconds. GPU event intervals overlap',
             'host evaluation time. Do not add these columns together.', '',
             '| Case | Scenario | Evaluation | Host preparation | GPU total | Dial responses | Event weights | Histograms | Histogram readback | Host likelihood |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fields = ('evaluationSeconds', 'hostPreparationSeconds', 'devicePropagationSeconds',
              'cachedDialStageSeconds', 'eventWeightsStageSeconds', 'histogramStageSeconds',
              'histogramReadbackSeconds', 'likelihoodHostSeconds')
    for case in cases:
        summary = json.loads((output / case / 'summary.json').read_text())
        for scenario, entry in summary['scenarios'].items():
            times = entry['timings']
            cells = [f'{times[key]["medianMs"]:.4f}' for key in fields]
            lines.append('| ' + ' | '.join([case, scenario, *cells]) + ' |')
    lines.extend(['', 'CPU stage durations in the responses/weights/histograms columns are host durations.',
                  'GPU total is zero for CPU. CUDA upload/reset duration is in the JSON reports.', '',
                  'Case summaries and coordinate costs include `medianResponseCandidates`.',
                  'For CUDA this counts dials visited by the response kernels, before transformed-input reuse.',
                  'Primed constant cached evaluations should have zero candidates.',
                  'Host sparse-selection time is `cachePreparationSeconds`, inside full evaluation time.', '',
                  'Inspect `local1-coordinate-costs.json` for the most expensive parameter changes.',
                  'Each case summary ranks CUDA stages in `gpuStageRanking`.',
                  '`comparison.json` checks every measured likelihood and each trial\'s final histogram',
                  '(contents and errors), and reports speedups against CPU when selected.', '',
                  'These are seeded propagation workloads, not the distribution of calls in a real fit.',
                  'The gap between evaluation and GPU time also contains host preparation, transfers,',
                  'likelihood/penalty work, launch overhead and synchronization. It is not all GPU wait.',
                  'No bandwidth, occupancy or hardware-counter claims are inferred from these timers.', ''])
    (output / 'report.md').write_text('\n'.join(lines))


def run_worker(args, case):
    directory = args.output if case == 'resident' else args.output / case
    if case != 'resident':
        directory.mkdir()
    log_path = directory / 'native.log'
    command = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:],
               '--worker', case, '--output', str(directory)]
    with log_path.open('w') as log:
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
                    progress(f'{case}: worker PID {child.pid} running; log {log_path}')
        except KeyboardInterrupt:
            child.terminate()
            child.wait()
            raise
        finally:
            reader.join()
    if code:
        raise RuntimeError(f'{case} exited with code {code}; see {log_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--override', nargs='*', default=[])
    parser.add_argument('--work-dir', type=Path, default=Path.cwd())
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES[:2]))
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--calls', type=int, default=30)
    parser.add_argument('--global-calls', type=int, default=10)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--step-scale', type=float, default=.1)
    parser.add_argument('--scenarios', nargs='+', choices=['constant', 'local1', 'local2', 'global'],
                        default=['constant', 'local1', 'local2', 'global'])
    parser.add_argument('--relative-tolerance', type=float, default=1e-9)
    parser.add_argument('--absolute-tolerance', type=float, default=1e-7)
    parser.add_argument('--output', type=Path,
                        default=Path(time.strftime('cuda-profiling-%Y%m%d-%H%M%S-') + str(os.getpid())))
    parser.add_argument('--execution-mode', choices=['resident', 'isolated'], default='resident')
    parser.add_argument('--worker', choices=(*CASES, 'resident'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.threads, args.calls, args.global_calls, args.trials) <= 0 or min(args.warmup, args.device_index) < 0:
        parser.error('Counts must be positive; warmup and device-index may be zero')
    for name in ('step_scale', 'relative_tolerance', 'absolute_tolerance'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} must be finite and positive')
    args.cases = list(dict.fromkeys(args.cases))
    args.scenarios = list(dict.fromkeys(args.scenarios))
    args.work_dir = args.work_dir.resolve()
    args.config = (args.work_dir / args.config).resolve()
    args.output = args.output.resolve()
    if args.worker:
        if args.worker == 'resident':
            measure_resident(args)
        else:
            measure(args)
        return
    if not args.config.is_file():
        parser.error(f'Configuration does not exist: {args.config}')
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(host=platform.node(), platform=platform.platform(), python=sys.executable,
                    allowedCpus=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                    environment={key: os.environ.get(key) for key in
                                 ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_ID', 'PYTHONPATH')})
    settings['gitCommit'] = subprocess.run(
        ['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'],
        capture_output=True, text=True, check=False).stdout.strip()
    try:
        settings['nvidiaSmi'] = subprocess.run(['nvidia-smi'], capture_output=True, text=True,
                                              timeout=15, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        settings['nvidiaSmi'] = None
    write_json(args.output / 'settings.json', settings)
    write_json(args.output / 'summary.json', {'complete': False})
    if args.execution_mode == 'resident':
        run_worker(args, 'resident')
    else:
        for case in args.cases:
            run_worker(args, case)
    compare(args.output, args.cases, args.scenarios, args.trials,
            args.relative_tolerance, args.absolute_tolerance)
    write_report(args.output, args.cases)
    write_json(args.output / 'summary.json', {
        'complete': True, 'cases': {case: json.loads((args.output / case / 'summary.json').read_text())
                                  for case in args.cases}})
    progress(f'SUCCESS: {args.output}')


if __name__ == '__main__':
    main()
