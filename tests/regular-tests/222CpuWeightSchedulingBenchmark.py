#!/usr/bin/env python3
"""Load once per thread count and compare fixed ranges with dynamic block factors.

Compare constant Asimov evaluations, one/two-coordinate local changes and
all-coordinate changes around the initialized point. These are reproducible
workloads, not actual Migrad or an MCMC chain. No minimization or Hesse is run.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True


def progress(message):
    print(f'[benchmark] {message}', flush=True)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def scenario_calls(args, scenario):
    return min(args.calls, args.global_calls) if scenario == 'global' else args.calls


class ParameterSequence:
    """Bounded, reproducible changes in the same coordinates used by the fit."""
    def __init__(self, engine, step_scale):
        self.entries = []
        sets = engine.getLikelihoodInterface().getModelPropagator().getParametersManager().getParameterSetsList()
        for parameter_set in sets:
            if not parameter_set.isEnabled():
                continue
            for parameter in parameter_set.getEffectiveParameterList():
                if not parameter.isEnabled() or parameter.isFixed() or parameter.isPenaltyDisabled():
                    continue
                center = parameter.getParameterValue()
                scale = parameter.getStdDevValue()
                if not math.isfinite(scale) or scale <= 0:
                    scale = parameter.getStepSize()
                if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
                    raise RuntimeError(f'No finite center/positive variation scale: {parameter.getFullTitle()}')
                self.entries.append((parameter, center, scale * step_scale,
                                     'eigen' if parameter_set.isEnableEigenDecomp() else 'original'))
        self.current = [center for _, center, _, _ in self.entries]

    def reset(self):
        for parameter, center, _, _ in self.entries:
            parameter.setParameterValue(center)
        self.current = [center for _, center, _, _ in self.entries]

    def manifest(self):
        return [{'index': i, 'name': p.getFullTitle(), 'center': center,
                 'proposalStdDev': scale, 'space': space}
                for i, (p, center, scale, space) in enumerate(self.entries)]

    def apply(self, scenario, rng, call, digest):
        if scenario == 'constant':
            return 0
        size = len(self.entries)
        count = {'local1': 1, 'local2': 2, 'global': size}[scenario]
        if size < count or not size:
            raise RuntimeError(f'{scenario} requires at least {max(1, count)} variable parameters; found {size}')
        # Cycle over coordinates rather than repeatedly timing one cheap dial.
        # Full coverage still requires enough calls for the parameter count.
        # Only the selected coordinates change; earlier coordinates retain their
        # small displacement until the trial ends (no hidden restoration calls).
        indices = range(size) if scenario == 'global' else [(call * count + j) % size for j in range(count)]
        for index in indices:
            parameter, center, scale, _ = self.entries[index]
            for _ in range(1000):
                candidate = center + rng.gauss(0., scale)
                if math.isfinite(candidate) and candidate != self.current[index] and parameter.isInDomain(candidate):
                    break
            else:
                raise RuntimeError(f'Cannot generate a distinct in-domain value: {parameter.getFullTitle()}')
            parameter.setParameterValue(candidate)
            self.current[index] = candidate
            digest.update(struct.pack('<Qd', index, candidate))
        return count


def measure_case(args, engine, metadata, tag, sequence, scenario):
    calls = scenario_calls(args, scenario)
    sequence.reset()
    engine.evaluateLikelihood()
    progress(f'{tag}: warming up ({args.warmup} evaluations)')
    warm_rng = random.Random(args.seed + 1000000)
    for call in range(args.warmup):
        sequence.apply(scenario, warm_rng, call, hashlib.sha256())
        engine.evaluateLikelihood()
    trials, raw = [], []
    for trial in range(args.trials):
        # Prime exactly the same center/cache before each measured sequence.
        sequence.reset()
        engine.evaluateLikelihood()
        rng, digest = random.Random(args.seed + trial), hashlib.sha256()
        progress(f'{tag}: trial {trial + 1}/{args.trials}, 0/{calls} evaluations')
        stages, caches, histograms, complete, updates, changed, likelihoods = [], [], [], [], [], [], []
        preparations, executions, candidates, linear = [], [], [], []
        input_transforms, shared_inputs = [], []
        last_progress = time.monotonic()
        for call in range(calls):
            update_start = time.perf_counter()
            changed.append(sequence.apply(scenario, rng, call, digest))
            updates.append(time.perf_counter() - update_start)
            start = time.perf_counter()
            engine.evaluateLikelihood()
            complete.append(time.perf_counter() - start)
            value = engine.getLikelihoodInterface().getLastLikelihood()
            if not math.isfinite(value):
                raise RuntimeError(f'{tag}: nonfinite likelihood at trial {trial + 1}, call {call + 1}')
            likelihoods.append(value)
            timing = engine.getBackendsManager().getLastTimingSummary()
            stages.append(timing.eventWeightsStageSeconds)
            caches.append(timing.cachedDialStageSeconds)
            preparations.append(timing.cachePreparationSeconds)
            executions.append(timing.cacheExecutionSeconds)
            candidates.append(timing.cacheCandidateCount)
            input_transforms.append(timing.cacheInputTransformCount)
            shared_inputs.append(timing.sharedCacheInputs)
            linear.append(timing.linearCacheTraversal)
            histograms.append(timing.histogramStageSeconds)
            if time.monotonic() - last_progress >= 5:
                progress(f'{tag}: trial {trial + 1}/{args.trials}, {call + 1}/{calls} evaluations')
                last_progress = time.monotonic()
        raw.append({'weightSeconds': stages, 'cacheSeconds': caches, 'histogramSeconds': histograms,
                    'evaluationSeconds': complete, 'parameterUpdateSeconds': updates,
                    'changedFitCoordinates': changed, 'likelihoods': likelihoods,
                    'cachePreparationSeconds': preparations, 'cacheExecutionSeconds': executions,
                    'cacheCandidates': candidates, 'linearCacheTraversal': linear,
                    'inputTransforms': input_transforms, 'sharedCacheInputs': shared_inputs})
        trials.append({'weightMedianMs': statistics.median(stages) * 1000,
                       'weightMeanMs': statistics.mean(stages) * 1000,
                       'weightP95Ms': sorted(stages)[int((len(stages) - 1) * .95)] * 1000,
                       'cacheMedianMs': statistics.median(caches) * 1000,
                       'cachePreparationMedianMs': statistics.median(preparations) * 1000,
                       'cacheExecutionMedianMs': statistics.median(executions) * 1000,
                       'cacheCandidateMedian': statistics.median(candidates),
                       'linearCacheCalls': sum(linear),
                       'inputTransformsMedian': statistics.median(input_transforms),
                       'sharedInputCalls': sum(shared_inputs),
                       'histogramMedianMs': statistics.median(histograms) * 1000,
                       'evaluationMedianMs': statistics.median(complete) * 1000,
                       'parameterUpdateMedianMs': statistics.median(updates) * 1000,
                       'changedFitCoordinates': sorted(set(changed)), 'sequenceDigest': digest.hexdigest()})
        write_json(args.output / f'{tag}.partial.json', {
            'complete': False, 'scenario': scenario, 'layout': metadata, 'trials': trials, 'raw': raw})
        progress(f"{tag}: trial {trial + 1}/{args.trials} finished; cache {trials[-1]['cacheMedianMs']:.3f} ms, reweight {trials[-1]['weightMedianMs']:.3f} ms")
    li = engine.getLikelihoodInterface()
    histogram = [(v.sumWeights, v.sqrtSumSqWeights)
                 for sample in li.getModelPropagator().getSampleSet().getSampleList()
                 for v in sample.getHistogram().getBinContentList()]
    return {'complete': True, 'scenario': scenario, 'seed': args.seed, 'stepScale': args.step_scale,
            'variableFitCoordinates': len(sequence.entries), 'actualScheduling': metadata['weightScheduling'],
            'workBlocks': metadata['weightBlockCount'], 'threads': metadata['threads'],
            'cacheInputGroups': metadata['cacheInputGroups'],
            'cacheInputStorageBytes': metadata['cacheInputStorageBytes'],
            'perDialInputStorageBytes': metadata['perDialInputStorageBytes'],
            'host': platform.node(), 'pid': os.getpid(), 'commit': metadata['commit'],
            'compilerOptimizationEnabled': metadata['compilerOptimizationEnabled'],
            'allowedCpus': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'events': metadata['events'], 'eventDialReferences': metadata['eventDialReferences'],
            'raw': raw, 'callsPerTrial': calls, 'trials': trials, 'histogram': histogram,
            'weightMedianMs': statistics.median(t['weightMedianMs'] for t in trials),
            'cacheMedianMs': statistics.median(t['cacheMedianMs'] for t in trials),
            'cachePreparationMedianMs': statistics.median(t['cachePreparationMedianMs'] for t in trials),
            'cacheExecutionMedianMs': statistics.median(t['cacheExecutionMedianMs'] for t in trials),
            'evaluationMedianMs': statistics.median(t['evaluationMedianMs'] for t in trials),
            'histogramMedianMs': statistics.median(t['histogramMedianMs'] for t in trials)}


def measure_suite(args):
    import GUNDAM
    threads = args.worker_threads
    if not hasattr(GUNDAM.BackendManager, 'setDiagnosticCpuInputEvaluation') or not hasattr(GUNDAM.ParameterSet, 'getEffectiveParameterList'):
        raise RuntimeError('Rebuild GUNDAM and select its Python module: runtime scheduling diagnostics are missing.')
    progress(f't{threads}: loading model once; GUNDAM module: {GUNDAM.__file__}')
    GUNDAM.setRuntimeWorkingDirectory(str(args.work_dir))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(threads)
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfig(str(args.config))
    if args.override:
        builder.override(args.override)
    config = json.loads(builder.getConfig().toString())['fitterEngineConfig']
    config.update(enablePca=False, scaleParStepWithChi2Response=False,
                  generateSamplePlots=False, generateOneSigmaPlots=False)
    # The benchmark uses cheap runtime layout diagnostics; no sampled probes
    # or OS snapshots run during the measurements.
    config['likelihoodInterfaceConfig']['backendManagerConfig'] = {
        'isEnabled': True, 'type': 'CPU', 'backendConfig': {'weightScheduling': 'Fixed'}}
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    load_start = time.monotonic()
    engine.initialize()
    progress(f't{threads}: initialization finished in {time.monotonic() - load_start:.1f} s; dataset retained for all cases')
    summary_path = args.output / 'summary.json'
    results = json.loads(summary_path.read_text()) if summary_path.exists() else []
    sequence = ParameterSequence(engine, args.step_scale)
    required = 2 if 'local2' in args.scenarios else int(any(s != 'constant' for s in args.scenarios))
    if len(sequence.entries) < required:
        raise RuntimeError(f'Requested scenarios require {required} variable fit coordinates; found {len(sequence.entries)}')
    write_json(args.output / f'parameters-t{threads}.json', sequence.manifest())
    progress(f't{threads}: {len(sequence.entries)} variable fit coordinates; scenarios: {args.scenarios}')
    references = {}
    if args.benchmark == 'inputs':
        layouts = [('Dynamic', 2, 'Auto', evaluation) for evaluation in ('PerDial', 'Shared')]
    elif args.benchmark == 'cache':
        layouts = [('Dynamic', 2, traversal, 'Shared') for traversal in ('Indexed', 'Auto')]
    else:
        layouts = [(mode, factor, 'Auto', 'Shared') for mode, factor in
                   [('Fixed', 2), *[('Dynamic', f) for f in dict.fromkeys(args.factors)]]]
    cases = [(scenario, *layout) for scenario in dict.fromkeys(args.scenarios) for layout in layouts]
    for round_index in range(args.rounds):
        ordered = cases if round_index % 2 == 0 else list(reversed(cases))
        for scenario, mode, factor, traversal, evaluation in ordered:
            tag = f'round{round_index + 1}-t{threads}-{scenario}-{mode}' + (f'-x{factor}' if mode == 'Dynamic' else '')
            if args.benchmark == 'cache':
                tag += f'-cache{traversal}'
            elif args.benchmark == 'inputs':
                tag += f'-inputs{evaluation}'
            progress(f'{tag}: rebuilding work ranges (events and cached responses retained)')
            engine.getBackendsManager().setDiagnosticCpuWeightScheduling(mode, factor)
            engine.getBackendsManager().setDiagnosticCpuCacheTraversal(traversal)
            metadata = json.loads(engine.getBackendsManager().setDiagnosticCpuInputEvaluation(evaluation))
            if args.benchmark == 'inputs':
                progress(f"{tag}: {metadata['cacheInputGroups']} shared input groups; "
                         f"input storage {metadata['cacheInputStorageBytes']} bytes "
                         f"(per-dial reference {metadata['perDialInputStorageBytes']} bytes)")
            if mode == 'Dynamic' and metadata['weightScheduling'] != 'Dynamic':
                raise RuntimeError(f'{tag}: too few events for the requested dynamic block count')
            count = metadata['weightBlockCount'] if mode == 'Dynamic' else threads
            progress(f'{tag}: {count} work ranges ({mode}); starting evaluations')
            result = measure_case(args, engine, metadata, tag, sequence, scenario)
            result.update(mode=mode, blocksPerThread=factor if mode == 'Dynamic' else None,
                          round=round_index + 1, pythonModule=GUNDAM.__file__,
                          cacheTraversal=traversal, inputEvaluation=evaluation, benchmark=args.benchmark)
            values = result['histogram']
            signature = [t['sequenceDigest'] for t in result['trials']]
            likelihoods = [v for trial in result['raw'] for v in trial['likelihoods']]
            if scenario not in references:
                references[scenario] = (values, signature, likelihoods)
            else:
                expected, expected_signature, expected_llh = references[scenario]
                histograms_agree = len(values) == len(expected) and all(
                    all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(actual, ref))
                    for actual, ref in zip(values, expected))
                llh_agree = len(likelihoods) == len(expected_llh) and all(
                    math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10) for a, b in zip(likelihoods, expected_llh))
                if signature != expected_signature or not histograms_agree or not llh_agree:
                    write_json(args.output / f'{tag}.json', result)
                    raise RuntimeError(f'{tag}: parameter sequence, likelihoods or histograms differ from reference')
            result['likelihoodsAgree'] = True
            result['histogramsAgree'] = True
            write_json(args.output / f'{tag}.json', result)
            (args.output / f'{tag}.partial.json').unlink(missing_ok=True)
            compact = {k: v for k, v in result.items() if k not in ('raw', 'histogram')}
            results.append(compact)
            for measured in results:
                baseline = next((r for r in results if r['round'] == measured['round']
                                 and r['threads'] == measured['threads'] and r['scenario'] == measured['scenario']
                                 and (r['inputEvaluation'] == 'PerDial' if args.benchmark == 'inputs' else
                                      r['cacheTraversal'] == 'Indexed' if args.benchmark == 'cache' else r['mode'] == 'Fixed')), None)
                if baseline:
                    if args.benchmark == 'inputs':
                        measured['cacheSpeedupVsPerDial'] = baseline['cacheMedianMs'] / measured['cacheMedianMs']
                        measured['evaluationSpeedupVsPerDial'] = baseline['evaluationMedianMs'] / measured['evaluationMedianMs']
                    elif args.benchmark == 'cache':
                        measured['cacheSpeedupVsIndexed'] = baseline['cacheMedianMs'] / measured['cacheMedianMs']
                        measured['evaluationSpeedupVsIndexed'] = baseline['evaluationMedianMs'] / measured['evaluationMedianMs']
                    else:
                        measured['weightSpeedupVsFixed'] = baseline['weightMedianMs'] / measured['weightMedianMs']
                        measured['evaluationSpeedupVsFixed'] = baseline['evaluationMedianMs'] / measured['evaluationMedianMs']
            write_json(summary_path, results)
            progress(f"{tag}: saved; reweight {result['weightMedianMs']:.3f} ms, evaluation {result['evaluationMedianMs']:.3f} ms; histograms agree")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=['weights', 'cache', 'inputs'], default='weights',
                        help='Compare weight factors, cache traversal, or PerDial/Shared input evaluation (Dynamic x2)')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--override', nargs='*', default=[])
    parser.add_argument('--work-dir', type=Path, default=Path.cwd())
    parser.add_argument('--threads', nargs='+', type=int, default=[8])
    parser.add_argument('--factors', nargs='+', type=int, default=[2, 5, 10])
    parser.add_argument('--scenarios', nargs='+', choices=['constant', 'local1', 'local2', 'global'],
                        default=['constant', 'local1', 'local2', 'global'])
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--step-scale', type=float, default=.1,
                        help='Proposal standard deviation as a fraction of prior sigma (step size for flat parameters)')
    parser.add_argument('--calls', type=int, default=100)
    parser.add_argument('--global-calls', type=int, default=20,
                        help='Maximum evaluations per trial for the costly global workload')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=1, help='Repeat cases, alternating forward and reverse order')
    parser.add_argument('--output', type=Path,
                        default=Path(time.strftime('cpu-weight-benchmark-%Y%m%d-%H%M%S-') + str(os.getpid())))
    parser.add_argument('--worker-threads', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min([*args.threads, *args.factors, args.calls, args.global_calls, args.warmup, args.trials, args.rounds,
            args.worker_threads or 2]) <= 0:
        parser.error('All counts must be positive')
    if not math.isfinite(args.step_scale) or args.step_scale <= 0:
        parser.error('--step-scale must be finite and strictly positive')
    if 1 in args.threads or args.worker_threads == 1:
        parser.error('Dynamic comparisons require at least two threads')
    args.work_dir = args.work_dir.resolve()
    args.config = (args.work_dir / args.config).resolve()
    args.output = args.output.resolve()
    if args.worker_threads:
        measure_suite(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    progress(f'Results: {args.output}')
    progress(f'{len(set(args.threads))} model load(s), one per thread count; factors/rounds reuse the same model')
    counts = {s: scenario_calls(args, s) for s in dict.fromkeys(args.scenarios)}
    layout_count = 2 if args.benchmark in ('cache', 'inputs') else 1 + len(set(args.factors))
    total_calls = sum(counts.values()) * layout_count * args.trials * args.rounds * len(set(args.threads))
    progress(f'Timed evaluations per trial: {counts}; total across cases: {total_calls} (plus warmup/cache priming)')
    write_json(args.output / 'settings.json', {
        'config': str(args.config), 'override': args.override, 'workDir': str(args.work_dir),
        'threads': args.threads, 'factors': args.factors, 'rounds': args.rounds,
        'calls': args.calls, 'globalCalls': args.global_calls, 'callsByScenario': counts,
        'warmup': args.warmup, 'trials': args.trials, 'host': platform.node(),
        'scenarios': args.scenarios, 'seed': args.seed, 'stepScale': args.step_scale,
        'benchmark': args.benchmark,
        'parameterSequence': 'identical seeded sequences per scenario across layouts/rounds'})
    for threads in dict.fromkeys(args.threads):
        log_path = args.output / f't{threads}.log'
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--benchmark', args.benchmark, '--config', str(args.config),
                   '--work-dir', str(args.work_dir), '--worker-threads', str(threads), '--calls', str(args.calls),
                   '--global-calls', str(args.global_calls),
                   '--warmup', str(args.warmup), '--trials', str(args.trials), '--rounds', str(args.rounds),
                   '--output', str(args.output), '--factors', *map(str, args.factors),
                   '--scenarios', *args.scenarios, '--seed', str(args.seed), '--step-scale', str(args.step_scale)]
        if args.override:
            command += ['--override', *args.override]
        environment = os.environ.copy()
        environment.pop('GUNDAM_CPU_DIAGNOSTIC_BLOCKS_PER_THREAD', None)
        progress(f't{threads}: starting worker; full native log: {log_path}')
        with log_path.open('w') as log:
            with subprocess.Popen(command, cwd=args.work_dir, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, env=environment) as child:
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
                        progress(f't{threads}: worker PID {child.pid} still running; detailed log: {log_path.name}')
                reader.join()
                if code:
                    raise RuntimeError(f't{threads}: worker exited with code {code}; see {log_path}')
    progress('SUCCESS: all cases agree. Summary: ' + str(args.output / 'summary.json'))


if __name__ == '__main__':
    main()
