#!/usr/bin/env python3
"""Compare serial/interleaved event products on one resident production model.

Replays identical effective-parameter sequences, alternates order, and checks
likelihoods and histograms exactly. No minimization or Hesse is run. --workers
changes only active weight workers; the resident pool/histogram workers stay fixed.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
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
progress, write_json = scheduling.progress, scheduling.write_json
FIELDS = ('cachedDialStageSeconds', 'eventWeightsStageSeconds', 'histogramStageSeconds',
          'likelihoodHostSeconds')


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
    calls = scheduling.scenario_calls(args, scenario, len(sequence.entries))
    last_progress = time.monotonic()
    for call in range(calls):
        changed = sequence.apply(scenario, rng, call, digest)
        start = time.perf_counter()
        engine.evaluateLikelihood()
        elapsed = time.perf_counter() - start
        likelihood = engine.getLikelihoodInterface().getLastLikelihood()
        if not math.isfinite(likelihood):
            raise RuntimeError(f'{scenario}: nonfinite likelihood at {call}')
        timing = manager.getLastTimingSummary()
        rows.append({'likelihood': likelihood, 'changedCoordinates': changed,
                     'seconds': {**{name: getattr(timing, name) for name in FIELDS},
                                 'evaluationSeconds': elapsed}})
        if time.monotonic() - last_progress >= 10:
            progress(f'{scenario}: {call + 1}/{calls}')
            last_progress = time.monotonic()
    manager.materialize('Histograms')
    histograms = [(value.sumWeights, value.sqrtSumSqWeights)
                  for sample in engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
                  for value in sample.getHistogram().getBinContentList()]
    return {'rows': rows, 'histograms': histograms, 'sequenceDigest': digest.hexdigest(),
            'timings': summarize(rows)}


def measure(args):
    import GUNDAM
    if not hasattr(GUNDAM.BackendManager, 'setDiagnosticCpuWeightInterleaving'):
        raise RuntimeError('Rebuild/select the module with the interleaving diagnostics')
    if sys.platform == 'darwin':
        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        images = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
        write_json(args.output / 'runtime.json', {
            'pythonModule': GUNDAM.__file__, 'images': images,
            'pythonModuleSha256': hashlib.sha256(Path(GUNDAM.__file__).read_bytes()).hexdigest()})
        if Path(GUNDAM.__file__).resolve() not in {Path(image).resolve() for image in images}:
            raise RuntimeError('DYLD_LIBRARY_PATH selected a different GUNDAM module; see runtime.json')
    engine = scheduling.load_engine(args, args.threads)
    manager = engine.getBackendsManager()
    manager.setDiagnosticCpuWeightScheduling('Dynamic', 2)
    sequence = scheduling.ParameterSequence(engine, args.step_scale)
    write_json(args.output / 'parameters.json', sequence.manifest())
    summary = {'complete': False, 'pairs': []}
    try:
        for round_index in range(args.rounds):
            for workers in args.workers:
                metadata = json.loads(manager.setDiagnosticCpuWeightWorkers(workers, False))
                write_json(args.output / f'model-w{workers}.json', metadata)
                for scenario_index, scenario in enumerate(args.scenarios):
                    for trial in range(args.trials):
                        modes = ['serial', 'interleaved']
                        if (round_index + trial + scenario_index) % 2:
                            modes.reverse()
                        pair = {}
                        tag = f'round{round_index + 1}-w{workers}-{scenario}-trial{trial + 1}'
                        for mode in modes:
                            manager.setDiagnosticCpuWeightInterleaving(mode == 'interleaved')
                            progress(f'{tag}-{mode}')
                            pair[mode] = evaluate(args, engine, sequence, scenario, trial)
                            write_json(args.output / f'{tag}-{mode}.json', pair[mode])
                        left, right = pair['serial'], pair['interleaved']
                        if left['sequenceDigest'] != right['sequenceDigest'] or left['histograms'] != right['histograms'] or (
                                [r['likelihood'] for r in left['rows']] != [r['likelihood'] for r in right['rows']]):
                            raise RuntimeError(f'{tag}: numerical results or parameter sequence differ')
                        entry = {'round': round_index + 1, 'workers': workers, 'scenario': scenario,
                                 'trial': trial + 1, 'exactMatch': True,
                                 'timings': {mode: result['timings'] for mode, result in pair.items()}}
                        for name in ('eventWeightsStageSeconds', 'evaluationSeconds'):
                            entry[name + 'Speedup'] = left['timings'][name]['medianMs'] / right['timings'][name]['medianMs']
                        summary['pairs'].append(entry)
                        write_json(args.output / 'summary.json', summary)
                        progress(f"{tag}: exact match; weight speedup {entry['eventWeightsStageSecondsSpeedup']:.3f}x")
    finally:
        manager.setDiagnosticCpuWeightInterleaving(True)
    summary['complete'] = True
    write_json(args.output / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--override', nargs='*', default=[])
    parser.add_argument('--work-dir', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--workers', nargs='+', type=int)
    parser.add_argument('--calls', type=int, default=60)
    parser.add_argument('--global-calls', type=int, default=10)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--step-scale', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--scenarios', nargs='+', choices=['constant', 'local1', 'local2', 'global'],
                        default=['constant', 'local1', 'local2', 'global'])
    args = parser.parse_args()
    args.workers = list(dict.fromkeys(args.workers or [args.threads]))
    if min(args.threads, args.calls, args.global_calls, args.trials, args.rounds) < 1 or args.warmup < 0:
        parser.error('Counts must be positive; warmup may be zero')
    if min(args.workers) < 1 or max(args.workers) > args.threads:
        parser.error('Workers must be between one and the resident thread count')
    if not math.isfinite(args.step_scale) or args.step_scale <= 0:
        parser.error('Step scale must be finite and positive')
    args.work_dir = args.work_dir.resolve()
    args.config = (args.work_dir / args.config).resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'settings.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    measure(args)


if __name__ == '__main__':
    main()
