#!/usr/bin/env python3
"""Replay Eb coordinates on a resident CPU model; retain exact output digests."""
import argparse
from array import array
import ctypes
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(values):
    return hashlib.sha256(array('d', values).tobytes()).hexdigest()


def progress(message):
    print('[Eb benchmark] ' + message, flush=True)


def run(args):
    import GUNDAM
    GUNDAM.setRuntimeWorkingDirectory(str(args.work_dir))
    GUNDAM.setNumberOfThreads(args.threads)
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setIsDebug(False)
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfig(str(args.work_dir / args.config))
    builder.override(args.override)
    config = json.loads(builder.getConfig().toString())['fitterEngineConfig']
    config.update(enablePca=False, scaleParStepWithChi2Response=False,
                  generateSamplePlots=False, generateOneSigmaPlots=False)
    backend = config['likelihoodInterfaceConfig'].setdefault('backendManagerConfig', {})
    backend.update(isEnabled=True, type='CPU')
    backend.setdefault('backendConfig', {})['profileOutputPath'] = ''
    save(args.output / 'resolved-config.json', config)
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    progress('Loading original Asimov data with ' + GUNDAM.__file__)
    start = time.perf_counter()
    engine.initialize()
    initialization = time.perf_counter() - start
    manager = engine.getBackendsManager()
    assert manager.hasBackend() and manager.getType() == 'CPU'
    manager.setEnableAutoMaterialize(False)
    runtime = {'module': GUNDAM.__file__, 'initializationSeconds': initialization,
               'moduleSha256': hashlib.sha256(Path(GUNDAM.__file__).read_bytes()).hexdigest()}
    if sys.platform == 'darwin':
        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        images = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
        runtime['images'] = images
        assert Path(GUNDAM.__file__).resolve() in {Path(image).resolve() for image in images}, 'Wrong GUNDAM image loaded'
    save(args.output / 'runtime.json', runtime)
    groups = engine.getLikelihoodInterface().getModelPropagator().getParametersManager().getParameterSetsList()
    entries = []
    for group in groups:
        if not group.isEnabled() or group.getName() != 'Cross-Section Systematics Eb':
            continue
        assert not group.isEnableEigenDecomp(), 'This replay uses the physical Eb coordinates'
        for parameter in group.getEffectiveParameterList():
            if not parameter.isEnabled() or parameter.isFixed():
                continue
            scale = parameter.getStdDevValue()
            if not math.isfinite(scale) or scale <= 0:
                scale = parameter.getStepSize()
            assert math.isfinite(scale) and scale > 0
            entries.append((parameter, parameter.getParameterValue(), scale))
    assert len(entries) == 5
    names = [p.getName() for p, _, _ in entries]
    save(args.output / 'parameters.json', [{'name': p.getFullTitle(), 'center': center, 'sigma': sigma}
                                         for p, center, sigma in entries])
    scenarios = {'constant': [], **{name: [i] for i, name in enumerate(names)}, 'all_Eb': list(range(len(entries)))}
    model = engine.getLikelihoodInterface().getModelPropagator()
    samples = model.getSampleSet().getSampleList()
    progress(f'Initialized in {initialization:.2f}s; {sum(len(s.getEventList()) for s in samples)} events')

    def reset():
        for parameter, center, _ in entries:
            parameter.setParameterValue(center)
        engine.evaluateLikelihood()

    def set_point(indices, call):
        for index in indices:
            parameter, center, sigma = entries[index]
            value = center + [-.2, .2, -.1, .1][call % 4] * sigma
            assert math.isfinite(value) and parameter.isInDomain(value)
            parameter.setParameterValue(value)

    def histogram():
        manager.materialize('Histograms')
        return digest(value for sample in samples for content in
                      [*sample.getHistogram().getBinContentList(), sample.getHistogram().getOutsideBinContent()]
                      for value in (content.sumWeights, content.sqrtSumSqWeights))

    seconds = ['cachedDialStageSeconds', 'eventWeightsStageSeconds', 'histogramStageSeconds',
               'variableStageSeconds', 'binningStageSeconds', 'likelihoodHostSeconds',
               'cachePreparationSeconds', 'cacheExecutionSeconds']
    counts = ['cacheCandidateCount', 'cacheInputTransformCount', 'rebinnedEventCount']
    rows, checks = {name: [] for name in scenarios}, {}
    order = list(scenarios)
    for trial in range(args.trials):
        for name in order[trial % len(order):] + order[:trial % len(order)]:
            reset()
            for call in range(args.warmup + args.calls):
                set_point(scenarios[name], call)
                start = time.perf_counter()
                engine.evaluateLikelihood()
                elapsed = time.perf_counter() - start
                likelihood = engine.getLikelihoodInterface().getLastLikelihood()
                assert math.isfinite(likelihood)
                if call < args.warmup:
                    continue
                timing = manager.getLastTimingSummary()
                rows[name].append({'trial': trial, 'call': call - args.warmup, 'likelihood': likelihood,
                    'seconds': {**{key: getattr(timing, key) for key in seconds}, 'evaluationSeconds': elapsed,
                                'hostPreparationSeconds': manager.getLastPreparationSeconds()},
                    'counts': {key: getattr(timing, key) for key in counts}})
            checks[f'{name}/{trial}'] = {'histogram': histogram()}
            # Full host copies and Python event traversal are outside timing.
            if trial == args.trials - 1:
                manager.materialize('EventState')
                manager.materialize('EventWeights')
                checks[f'{name}/{trial}']['events'] = digest(value for sample in samples
                    for event in sample.getEventList() for value in
                    (event.getEventWeight(), event.getVariables().fetchVariable('Pmu').getVarAsDouble(),
                     float(event.getIndices().bin)))
            save(args.output / 'rows.json', rows)
            save(args.output / 'checks.json', checks)
            progress(f'Trial {trial + 1}/{args.trials}: {name}: '
                     f'{statistics.median(r["seconds"]["evaluationSeconds"] for r in rows[name])*1000:.3f} ms')
    reset()
    summary = {name: {'calls': len(items), 'timings': {
        field: {'medianMs': statistics.median(r['seconds'][field] for r in items)*1000,
                'p95Ms': sorted(r['seconds'][field] for r in items)[int(.95*(len(items)-1))]*1000}
        for field in items[0]['seconds']}, 'medianCounts': {
            field: statistics.median(r['counts'][field] for r in items) for field in counts}}
        for name, items in rows.items()}
    save(args.output / 'summary.json', summary)
    save(args.output / 'status.json', {'complete': True})
    progress('Complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('ndUpCcInc.yaml'))
    parser.add_argument('--override', nargs='+', required=True)
    parser.add_argument('--output', type=Path,
                        default=Path(f'benchmark-{Path(__file__).stem}-{time.strftime("%Y%m%d%H%M%S")}'),
                        help='Output directory (default: ./benchmark-<script-name>-YYYYMMDDHHmmss)')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--calls', type=int, default=160)
    parser.add_argument('--warmup', type=int, default=8)
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    args.output = args.output.resolve()
    assert min(args.threads, args.trials, args.calls) > 0 and args.warmup >= 0
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'settings.json', {key: str(value) if isinstance(value, Path) else value
                                       for key, value in vars(args).items()})
    run(args)


if __name__ == '__main__':
    main()
