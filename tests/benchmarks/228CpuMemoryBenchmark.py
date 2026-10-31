#!/usr/bin/env python3
"""Measure production CPU model memory and replay deterministic likelihoods.

Run before/after in separate processes with matching input selections. Loading,
memory snapshots and histogram copies are outside evaluation timers. No fit or
Hesse is run. A single initial CPU profile supplies buffer sizes/capacities.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import random
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    'scheduling', Path(__file__).with_name('222CpuWeightSchedulingBenchmark.py'))
scheduling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduling)
write_json, progress = scheduling.write_json, scheduling.progress


def summarize(rows):
    return {name: {'medianMs': statistics.median(row['seconds'][name] for row in rows) * 1000,
                   'p95Ms': sorted(row['seconds'][name] for row in rows)[int(.95 * (len(rows) - 1))] * 1000}
            for name in rows[0]['seconds']}


def evaluate(args, engine, sequence, scenario, trial):
    manager = engine.getBackendsManager()
    sequence.reset()
    engine.evaluateLikelihood()
    rng, digest = random.Random(args.seed + 1000000), hashlib.sha256()
    for call in range(args.warmup):
        sequence.apply(scenario, rng, call, digest)
        engine.evaluateLikelihood()
    sequence.reset()
    engine.evaluateLikelihood()
    rng, digest, rows = random.Random(args.seed + trial), hashlib.sha256(), []
    for call in range(scheduling.scenario_calls(args, scenario, len(sequence.entries))):
        changed = sequence.apply(scenario, rng, call, digest)
        start = time.perf_counter()
        engine.evaluateLikelihood()
        elapsed = time.perf_counter() - start
        likelihood = engine.getLikelihoodInterface().getLastLikelihood()
        if not math.isfinite(likelihood):
            raise RuntimeError(f'{scenario}: nonfinite likelihood at {call}')
        timing = manager.getLastTimingSummary()
        fields = ('cachedDialStageSeconds', 'eventWeightsStageSeconds', 'histogramStageSeconds',
                  'likelihoodHostSeconds')
        rows.append({'likelihood': likelihood, 'changedCoordinates': changed,
                     'seconds': {**{name: getattr(timing, name) for name in fields},
                                 'evaluationSeconds': elapsed}})
    manager.materialize('Histograms')
    histograms = [(value.sumWeights, value.sqrtSumSqWeights)
                  for sample in engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
                  for value in sample.getHistogram().getBinContentList()]
    return {'rows': rows, 'histograms': histograms, 'sequenceDigest': digest.hexdigest(),
            'timings': summarize(rows)}


def compare(reference, output):
    def read(path, name):
        return json.loads((path / name).read_text())
    if read(reference, 'parameters.json') != read(output, 'parameters.json'):
        raise RuntimeError('Reference parameter manifests differ')
    configs = [read(path, 'resolved-config.json') for path in (reference, output)]
    for config in configs:
        config['likelihoodInterfaceConfig']['backendManagerConfig']['options'][0].pop('profileOutputPath')
    if configs[0] != configs[1]:
        raise RuntimeError('Reference model configurations differ')
    before, after = read(reference, 'summary.json'), read(output, 'summary.json')
    if not before['complete'] or not after['complete']:
        raise RuntimeError('Cannot compare incomplete runs')
    settings = read(output, 'settings.json')
    previous_settings = read(reference, 'settings.json')
    for field in ('threads', 'calls', 'global_calls', 'trials', 'warmup', 'step_scale', 'seed', 'scenarios'):
        if settings[field] != previous_settings[field]:
            raise RuntimeError(f'Reference benchmark setting differs: {field}')
    for scenario in settings['scenarios']:
        for trial in range(1, settings['trials'] + 1):
            name = f'{scenario}-trial{trial}.json'
            left, right = read(reference, name), read(output, name)
            if (left['sequenceDigest'] != right['sequenceDigest'] or left['histograms'] != right['histograms']
                    or [r['likelihood'] for r in left['rows']] != [r['likelihood'] for r in right['rows']]):
                raise RuntimeError(f'Numerical results or parameter sequence differ: {name}')
    result = {'reference': str(reference), 'exactMatch': True,
              'initializationSeconds': [before['initializationSeconds'], after['initializationSeconds']],
              'memory': {'before': read(reference, 'memory.json'), 'after': read(output, 'memory.json')},
              'scenarios': {name: {'before': before['scenarios'][name], 'after': after['scenarios'][name]}
                            for name in settings['scenarios']}}
    write_json(output / 'comparison.json', result)
    progress('Reference likelihoods, histograms and parameter sequences match exactly')


def memory():
    result = {'peakRssBytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
              (1 if sys.platform == 'darwin' else 1024),
              'rssBytes': int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())])) * 1024}
    if sys.platform == 'darwin':
        class Statistics(ctypes.Structure):
            _fields_ = [('blocksInUse', ctypes.c_uint), ('sizeInUse', ctypes.c_size_t),
                        ('maxSizeInUse', ctypes.c_size_t), ('sizeAllocated', ctypes.c_size_t)]
        allocator = ctypes.CDLL(None)
        allocator.malloc_zone_statistics.argtypes = [ctypes.c_void_p, ctypes.POINTER(Statistics)]
        allocator.malloc_zone_statistics.restype = None
        stats = Statistics()
        allocator.malloc_zone_statistics(None, ctypes.byref(stats))
        result.update(allocatorInUseBytes=stats.sizeInUse, allocatorReservedBytes=stats.sizeAllocated)
    return result


def measure(args):
    import GUNDAM
    runtime = {'pythonModule': GUNDAM.__file__,
               'pythonModuleSha256': hashlib.sha256(Path(GUNDAM.__file__).read_bytes()).hexdigest()}
    if sys.platform == 'darwin':
        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        runtime['images'] = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
        if Path(GUNDAM.__file__).resolve() not in {Path(p).resolve() for p in runtime['images']}:
            raise RuntimeError('DYLD_LIBRARY_PATH substituted another GUNDAM module')
    write_json(args.output / 'runtime.json', runtime)
    progress(f'Module: {GUNDAM.__file__}')
    GUNDAM.setRuntimeWorkingDirectory(str(args.work_dir))
    GUNDAM.setNumberOfThreads(args.threads)
    GUNDAM.setLightOutputMode(True)
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfig(str(args.config))
    if args.override:
        builder.override(args.override)
    config = json.loads(builder.getConfig().toString())['fitterEngineConfig']
    config.update(enablePca=False, scaleParStepWithChi2Response=False,
                  generateSamplePlots=False, generateOneSigmaPlots=False)
    backend = config['likelihoodInterfaceConfig'].setdefault('backendManagerConfig', {})
    backend.pop('type', None)
    backend.update(isEnabled=True, useBackend='CPU')
    options = next((entry for entry in backend.setdefault('options', []) if entry['name'] == backend['useBackend']), None)
    if options is None:
        options = {'name': backend['useBackend']}
        backend['options'].append(options)
    options.update(
        profileOutputPath=str(args.output / 'cpu-profile.jsonl'), profileSampleInterval=100,
        profileMaxSamples=1)
    write_json(args.output / 'resolved-config.json', config)
    snapshots = {'beforeLoading': memory()}
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    start = time.monotonic()
    progress('Loading full Asimov model')
    engine.initialize()
    initialization_seconds = time.monotonic() - start
    snapshots['initialized'] = memory()
    write_json(args.output / 'memory.json', snapshots)
    manager = engine.getBackendsManager()
    manager.setEnableAutoMaterialize(False)
    engine.evaluateLikelihood()
    profile = args.output / f'cpu-profile-{os.getpid()}.jsonl'
    metadata = json.loads(profile.open().readline())
    write_json(args.output / 'model.json', metadata)
    progress(f"Initialized in {initialization_seconds:.1f}s: {metadata['events']} events, "
             f"{metadata['uniqueDials']} unique dials, allocator={snapshots['initialized'].get('allocatorInUseBytes')}")
    sequence = scheduling.ParameterSequence(engine, args.step_scale)
    write_json(args.output / 'parameters.json', sequence.manifest())
    summary = {'complete': False, 'initializationSeconds': initialization_seconds, 'scenarios': {}}
    for scenario in args.scenarios:
        trials = []
        for trial in range(args.trials):
            progress(f'{scenario}: trial {trial + 1}/{args.trials}')
            result = evaluate(args, engine, sequence, scenario, trial)
            write_json(args.output / f'{scenario}-trial{trial + 1}.json', result)
            trials.append(result)
            summary['scenarios'][scenario] = summarize(
                [row for t in trials for row in t['rows']])
            write_json(args.output / 'summary.json', summary)
    snapshots['afterEvaluations'] = memory()
    write_json(args.output / 'memory.json', snapshots)
    summary['complete'] = True
    write_json(args.output / 'summary.json', summary)
    if args.reference is not None:
        compare(args.reference.resolve(), args.output)
    progress('Finished')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--override', nargs='*', default=[])
    parser.add_argument('--work-dir', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path,
                        default=Path(f'benchmark-{Path(__file__).stem}-{time.strftime("%Y%m%d%H%M%S")}'),
                        help='Output directory (default: ./benchmark-<script-name>-YYYYMMDDHHmmss)')
    parser.add_argument('--reference', type=Path, help='Check exact numerical results against an earlier run')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--calls', type=int, default=60)
    parser.add_argument('--global-calls', type=int, default=10)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--step-scale', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--scenarios', nargs='+', choices=['constant', 'local1', 'local2', 'global'],
                        default=['constant', 'local1', 'global'])
    args = parser.parse_args()
    if min(args.threads, args.calls, args.global_calls, args.trials) < 1 or args.warmup < 0:
        parser.error('Counts must be positive; warmup may be zero')
    if not math.isfinite(args.step_scale) or args.step_scale <= 0:
        parser.error('Step scale must be finite and positive')
    args.work_dir = args.work_dir.resolve()
    args.config = (args.work_dir / args.config).resolve()
    args.output = args.output.resolve()
    if args.reference is not None:
        args.reference = args.reference.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'settings.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    measure(args)


if __name__ == '__main__':
    main()
