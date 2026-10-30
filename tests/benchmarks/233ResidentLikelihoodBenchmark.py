#!/usr/bin/env python3
"""Compare fit-style CPU/MPS evaluations on one resident production model.

No minimization or Hesse. Identical seeded coordinate sequences, warmup,
rotating backend order and separate synchronized Metal stage probes. Output
materialization and parameter setters are excluded from evaluation timings.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    'scheduling', Path(__file__).with_name('222CpuWeightSchedulingBenchmark.py'))
scheduling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduling)

SECONDS = ('propagationWallSeconds', 'parameterUploadSeconds', 'externalWeightUploadSeconds',
           'cachedDialStageSeconds', 'eventWeightsStageSeconds', 'histogramStageSeconds',
           'likelihoodHostSeconds', 'likelihoodDeviceSeconds', 'commandEncodeSeconds',
           'deviceWaitSeconds', 'devicePropagationSeconds', 'cachePreparationSeconds',
           'cacheExecutionSeconds', 'histogramReadbackSeconds', 'binningStageSeconds')
COUNTS = ('histogramReadbackBytes', 'eventWeightReadbackBytes', 'eventStateReadbackBytes',
          'likelihoodReadbackBytes', 'rebinnedEventCount', 'cacheCandidateCount')


def stats(values):
    ordered = sorted(values)
    return {'medianMs': statistics.median(values)*1000,
            'p95Ms': ordered[int(.95*(len(ordered)-1))]*1000,
            'meanMs': statistics.mean(values)*1000}


def save(path, obj):
    scheduling.write_json(path, obj)


def run(args):
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
    likelihood = config.setdefault('likelihoodInterfaceConfig', {})
    # OA2024 still uses the legacy frontend location. Leave its propagator
    # intact so GUNDAM handles relocation, and preserve its likelihood options.
    legacy = config.get('propagatorConfig', {})
    legacy_samples = legacy.get('sampleSetConfig', legacy.get('fitSampleSetConfig', {}))
    legacy_llh = legacy.get('llhConfig', legacy_samples.get('llhConfig', {}))
    likelihood.setdefault('jointProbabilityConfig', dict(legacy_llh))
    if args.likelihood == 'PoissonLLH':
        likelihood['jointProbabilityConfig'] = {'type': args.likelihood,
            'ignoreBinsWithZeroPredictionAtPrior': True}
    else:
        likelihood.setdefault('jointProbabilityConfig', {})['type'] = args.likelihood
    likelihood['backendManagerConfig'] = {'isEnabled': True, 'type': 'CPU'}
    save(args.output / 'resolved-config.json', config)
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    interface = engine.getLikelihoodInterface()
    interface.setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    start = time.perf_counter()
    scheduling.progress('Loading once: ' + str(args.config))
    engine.initialize()
    init_seconds = time.perf_counter()-start
    manager = engine.getBackendsManager()
    manager.setEnableAutoMaterialize(False)
    sequence = scheduling.ParameterSequence(engine, args.step_scale)
    samples = interface.getModelPropagator().getSampleSet().getSampleList()
    metadata = {'initializationSeconds': init_seconds, 'threads': args.threads,
                'events': sum(len(s.getEventList()) for s in samples),
                'bins': sum(len(s.getHistogram().getBinContentList()) for s in samples),
                'fitCoordinates': len(sequence.entries), 'module': GUNDAM.__file__,
                'moduleSha256': hashlib.sha256(Path(GUNDAM.__file__).read_bytes()).hexdigest(),
                'platform': platform.platform(), 'python': sys.executable,
                'likelihood': args.likelihood}
    save(args.output / 'model.json', metadata)
    save(args.output / 'parameters.json', sequence.manifest())
    scheduling.progress(f'Loaded {metadata}; starting measurements')
    cases = ['CPU'] if args.cpu_only else ['CPU', 'MPS-cachedResponses', 'MPS-recomputeResponses']
    report = {'model': metadata, 'complete': False, 'trials': [], 'probes': []}
    counts = {'constant': args.calls, 'local1': max(args.calls, len(sequence.entries)),
              'local2': max(args.calls, (len(sequence.entries)+1)//2), 'global': args.global_calls}

    def replace(case):
        if case == 'CPU':
            manager.replaceBackend('CPU', GUNDAM.JsonType('{}'))
        else:
            mode = case.removeprefix('MPS-')
            manager.replaceBackend('MPS', GUNDAM.JsonType(json.dumps({'propagationMode': mode})))
            diagnostics = json.loads(manager.getDiagnosticMps())
            if diagnostics.get('likelihoodExecution') != 'device':
                raise RuntimeError('Likelihood is not device resident: ' + str(diagnostics))
            save(args.output / (case+'-model.json'), diagnostics)
        manager.setEnableAutoMaterialize(False)

    def evaluate(case, scenario, count, seed, profiling=False):
        sequence.reset()
        engine.evaluateLikelihood()
        if case != 'CPU':
            manager.setDiagnosticMpsStageProfiling(profiling)
        rng, digest = random.Random(seed), hashlib.sha256()
        rows = []
        try:
            for call in range(count):
                logical_call = call if not profiling else call * max(1, len(sequence.entries)//count)
                sequence.apply(scenario, rng, logical_call, digest)
                start = time.perf_counter()
                engine.evaluateLikelihood()
                elapsed = time.perf_counter()-start
                value = interface.getLastLikelihood()
                if not math.isfinite(value):
                    raise RuntimeError(f'Nonfinite likelihood: {case}/{scenario}/{call}')
                timing = manager.getLastTimingSummary()
                row = {'evaluationSeconds': elapsed, 'likelihood': value,
                       'statLikelihood': interface.getLastStatLikelihood(),
                       'seconds': {name: getattr(timing, name) for name in SECONDS},
                       'counts': {name: getattr(timing, name) for name in COUNTS}}
                if case != 'CPU':
                    if row['counts']['histogramReadbackBytes'] or row['counts']['eventWeightReadbackBytes']:
                        raise RuntimeError('Unexpected histogram/weight readback during benchmark')
                    if row['counts']['likelihoodReadbackBytes'] != 8:
                        raise RuntimeError('Expected scalar-only likelihood readback')
                if profiling:
                    row['gpuStages'] = json.loads(manager.getDiagnosticMpsStageGpuSeconds())
                rows.append(row)
        finally:
            if case != 'CPU':
                manager.setDiagnosticMpsStageProfiling(False)
        return {'case': case, 'scenario': scenario, 'digest': digest.hexdigest(), 'rows': rows,
                'timing': stats([r['evaluationSeconds'] for r in rows]),
                'stages': {name: stats([r['seconds'][name] for r in rows]) for name in SECONDS}}

    for trial in range(args.trials):
        order = cases[trial % len(cases):] + cases[:trial % len(cases)]
        for case in order:
            sequence.reset()
            replace(case)
            for scenario, count in counts.items():
                scheduling.progress(f'trial {trial+1}/{args.trials}, {case}/{scenario}: {count} calls')
                evaluate(case, scenario, args.warmup, args.seed+100000)
                result = evaluate(case, scenario, count, args.seed+trial)
                result['trial'] = trial
                report['trials'].append(result)
                save(args.output / 'results.json', report)
        # Compare every timed parameter point with the CPU on the same trial.
        for scenario in counts:
            subset = [r for r in report['trials'] if r['trial'] == trial and r['scenario'] == scenario]
            cpu = next(r for r in subset if r['case'] == 'CPU')
            for other in subset:
                if other['digest'] != cpu['digest']:
                    raise RuntimeError('Parameter sequence differs between backends')
                errors = [abs(a['statLikelihood']-b['statLikelihood']) for a,b in zip(cpu['rows'],other['rows'])]
                other['maxStatLikelihoodDifferenceFromCpu'] = max(errors)
                for a,b in zip(cpu['rows'],other['rows']):
                    if not math.isclose(a['statLikelihood'], b['statLikelihood'], rel_tol=1e-8, abs_tol=2e-7):
                        raise RuntimeError(f'CPU/MPS likelihood mismatch: {a["statLikelihood"]}, {b["statLikelihood"]}')
        save(args.output / 'results.json', report)

    # Materialization is measured separately, after the ordinary passes.
    for case in cases:
        replace(case)
        sequence.reset()
        engine.evaluateLikelihood()
        start = time.perf_counter()
        manager.materialize('Histograms')
        seconds = time.perf_counter()-start
        bins = [(b.sumWeights, b.sqrtSumSqWeights) for s in samples for b in s.getHistogram().getBinContentList()]
        save(args.output / (case+'-histogram.json'), {'seconds': seconds, 'bins': bins})
        if case == 'CPU' or not args.probe_calls:
            continue
        for scenario in ('constant', 'local1', 'global'):
            scheduling.progress(f'Separate stage probes: {case}/{scenario}')
            report['probes'].append(evaluate(case, scenario, args.probe_calls, args.seed, profiling=True))
            save(args.output / 'results.json', report)
    report['complete'] = True
    save(args.output / 'results.json', report)
    scheduling.progress('Benchmark complete: ' + str(args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--override', nargs='*', default=[])
    parser.add_argument('--likelihood', choices=['PoissonLLH', 'BarlowLLH_BANFF_OA2021'], default='BarlowLLH_BANFF_OA2021')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--calls', type=int, default=100)
    parser.add_argument('--global-calls', type=int, default=40)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--probe-calls', type=int, default=8)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--step-scale', type=float, default=.1)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    args.config = (args.work_dir/args.config).resolve()
    args.output = args.output.resolve()
    if min(args.threads, args.calls, args.global_calls, args.trials, args.warmup) <= 0 or args.probe_calls < 0:
        parser.error('Counts must be positive; probe-calls may be zero')
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'settings.json', {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    run(args)


if __name__ == '__main__':
    main()
