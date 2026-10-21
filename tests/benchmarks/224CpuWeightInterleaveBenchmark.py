#!/usr/bin/env python3
"""Compare scalar, two-event and four-event CPU weight kernels on one loaded model.

All variants retain the same buffers, worker pool and Dynamic x2 scheduler.
The input point is constant and the response cache primed. No fit/HESSE is run.
"""
import importlib.util
import json
import math
import os
import platform
import struct
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    'scaling', Path(__file__).with_name('223CpuWeightScalingBenchmark.py'))
scaling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaling)
progress, write_json = scaling.progress, scaling.write_json


def measure(args):
    import GUNDAM
    if not hasattr(GUNDAM.BackendManager, 'setDiagnosticCpuWeightInterleave'):
        raise RuntimeError('Rebuild/install GUNDAM: CPU interleaved kernels are missing.')
    engine = scaling.scheduling.load_engine(args, args.threads)
    manager = engine.getBackendsManager()
    manager.setDiagnosticCpuWeightScheduling('Dynamic', 2)
    metadata = json.loads(manager.setDiagnosticCpuWeightInterleave(1))
    if not metadata['allWeightResponsesCached']:
        raise RuntimeError('This experiment requires fully cached event responses; mixed models retain the scalar kernel.')
    engine.evaluateLikelihood()
    reference_llh = engine.getLikelihoodInterface().getLastLikelihood()
    manager.materialize('Histograms')

    def histogram_bits():
        return b''.join(struct.pack('=dd', value.sumWeights, value.sqrtSumSqWeights)
                        for sample in engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
                        for value in sample.getHistogram().getBinContentList())

    reference_histogram = histogram_bits()
    events, chunk_events = metadata['events'], 65536
    reference_path = args.output / 'reference-weights.bin'
    # Disk-backed baseline keeps exact comparison bounded to two chunks, without
    # retaining a second full-size event-weight array in either backend or Python.
    with reference_path.open('wb') as baseline:
        for offset in range(0, events, chunk_events):
            baseline.write(manager.getDiagnosticCpuEventWeightChunk(offset, min(chunk_events, events - offset)))
    write_json(args.output / 'model.json', {
        'model': metadata, 'host': platform.node(), 'pid': os.getpid(),
        'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'autoMaterializeDuringTiming': False, 'comparisonChunkEvents': chunk_events,
        'policy': 'constant point; same pool and Dynamic x2 ranges; kernel width only changes'})
    results = []
    for round_index in range(args.rounds):
        order = args.lanes if round_index % 2 == 0 else list(reversed(args.lanes))
        for lanes in order:
            tag = f'round{round_index + 1}-lanes{lanes}'
            layout = json.loads(manager.setDiagnosticCpuWeightInterleave(lanes))
            if layout['weightInterleave'] != lanes:
                raise RuntimeError('Requested kernel was not selected.')
            progress(f'{tag}: warmup; model, response cache and allocation retained')
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
                        raise RuntimeError('Constant-point response cache is not warm.')
                    if not math.isclose(engine.getLikelihoodInterface().getLastLikelihood(), reference_llh,
                                        rel_tol=1e-10, abs_tol=1e-10):
                        raise RuntimeError('Kernel changed the likelihood.')
                raw.append(times)
                write_json(args.output / f'{tag}.partial.json', {'complete': False, 'raw': raw})
                progress(f"{tag}: trial {trial + 1}/{args.trials}, weights {scaling.stats(times['weights'])['medianMs']:.3f} ms")
            # Comparisons and host publication are outside the timing loop.
            with reference_path.open('rb') as baseline:
                for offset in range(0, events, chunk_events):
                    count = min(chunk_events, events - offset)
                    actual = manager.getDiagnosticCpuEventWeightChunk(offset, count)
                    if actual != baseline.read(count * 8):
                        raise RuntimeError(f'{tag}: event weight bits differ in chunk starting at {offset}')
            manager.materialize('Histograms')
            if histogram_bits() != reference_histogram:
                raise RuntimeError(f'{tag}: histogram sum/squared-sum bits differ')
            flattened = {key: [v for trial in raw for v in trial[key]] for key in raw[0]}
            result = {'complete': True, 'round': round_index + 1, 'lanes': lanes,
                      'poolThreads': args.threads, 'layout': layout, 'raw': raw,
                      'timings': {key: scaling.stats(values) for key, values in flattened.items()},
                      'likelihoodsAgree': True, 'weightsExactlyAgree': True, 'histogramsExactlyAgree': True}
            write_json(args.output / f'{tag}.json', result)
            (args.output / f'{tag}.partial.json').unlink(missing_ok=True)
            results.append({k: v for k, v in result.items() if k not in ('raw', 'layout')})
            for item in results:
                baseline = next((r for r in results if r['round'] == item['round'] and r['lanes'] == 1), None)
                if baseline:
                    item['weightSpeedupVsScalar'] = baseline['timings']['weights']['medianMs'] / item['timings']['weights']['medianMs']
            write_json(args.output / 'summary.json', results)
    reference_path.unlink()
    progress(f'SUCCESS: all kernels preserve weight/histogram bits and likelihoods. Results: {args.output}')


if __name__ == '__main__':
    scaling.main(measure_function=measure, interleaving=True, description=__doc__)
