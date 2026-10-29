#!/usr/bin/env python3
"""Measure resident CPU propagation with targeted variable-dial changes.

Run from the production input directory with its Python environment. No fit or
Hesse is run. Dataset selections, binning and backend settings are preserved.
"""
import argparse
import ctypes
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from array import array

sys.dont_write_bytecode = True

SECONDS = ('cachedDialStageSeconds', 'eventWeightsStageSeconds',
           'variableStageSeconds', 'binningStageSeconds', 'histogramStageSeconds',
           'likelihoodHostSeconds', 'cachePreparationSeconds', 'cacheExecutionSeconds')
COUNTS = ('rebinnedEventCount', 'binCandidateCount', 'binContributionCount')


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def progress(message):
    print(f'[benchmark] {message}', flush=True)


def measure(args):
    import GUNDAM
    missing = [field for field in (*SECONDS, *COUNTS)
               if not hasattr(GUNDAM.BackendTimingSummary, field)]
    if missing:
        raise RuntimeError(f'Select/rebuild the matching Python module; missing fields: {missing}')
    progress(f'Python module: {GUNDAM.__file__}')
    GUNDAM.setRuntimeWorkingDirectory(str(args.work_dir))
    GUNDAM.setNumberOfThreads(args.threads)
    GUNDAM.setLightOutputMode(True)
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfig(str(args.config))
    if args.override:
        builder.override(args.override)
    config = json.loads(builder.getConfig().toString())['fitterEngineConfig']
    backend = config['likelihoodInterfaceConfig'].setdefault('backendManagerConfig', {})
    backend.update(isEnabled=True, type='CPU')
    # Ordinary stage timers suffice; exclude sampled OS probes from this pass.
    backend.setdefault('backendConfig', {})['profileOutputPath'] = ''
    config.update(generateSamplePlots=False, generateOneSigmaPlots=False)
    save(args.output / 'resolved-fitter-config.json', config)
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    progress('Loading Asimov model once; no minimization or Hesse')
    engine.initialize()
    metadata = {'pythonModule': GUNDAM.__file__,
                'pythonModuleSha256': hashlib.sha256(Path(GUNDAM.__file__).read_bytes()).hexdigest()}
    if sys.platform == 'darwin':
        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        metadata['loadedImages'] = [dyld._dyld_get_image_name(i).decode()
                                   for i in range(dyld._dyld_image_count())]
    save(args.output / 'runtime.json', metadata)
    if 'loadedImages' in metadata and Path(GUNDAM.__file__).resolve() not in {
            Path(image).resolve() for image in metadata['loadedImages']}:
        raise RuntimeError('macOS loaded a different GUNDAM image than Python requested. '
                           'Remove the installed GUNDAM directory from DYLD_LIBRARY_PATH; see runtime.json.')
    manager = engine.getBackendsManager()
    if not manager.hasBackend() or manager.getType() != 'CPU':
        raise RuntimeError('Expected an active CPU backend')
    manager.setEnableAutoMaterialize(False)
    sets = engine.getLikelihoodInterface().getModelPropagator().getParametersManager().getParameterSetsList()
    matches = [p for group in sets if group.isEnabled()
               for p in group.getEffectiveParameterList()
               if p.getName() == args.parameter and p.isEnabled() and not p.isFixed()]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one active effective parameter named {args.parameter}; found {len(matches)}. '
                           'For stress, apply pmuScaleBinningTest.yaml before pmuScaleBinningStressTest.yaml.')
    parameter = matches[0]
    center, sigma = parameter.getParameterValue(), parameter.getStdDevValue()
    values = [center + offset * sigma for offset in args.sigma_offsets]
    if not math.isfinite(sigma) or sigma <= 0 or any(
            not math.isfinite(value) or not parameter.isInDomain(value) for value in values):
        raise RuntimeError(f'Invalid/out-of-domain parameter points: {values}')
    if any(left == right for left, right in zip(values, values[1:] + values[:1])):
        raise RuntimeError('Each successive point, including cycle wrap, must change the parameter')
    save(args.output / 'points.json', {'parameter': parameter.getFullTitle(),
                                     'center': center, 'sigma': sigma, 'values': values})
    summary = {'complete': False, 'scenarios': {}}
    # Control: repeat identical inputs. Variable: change only the requested dial.
    try:
        for scenario in ('constant', 'variable'):
            trials = []
            for trial in range(args.trials):
                parameter.setParameterValue(center)
                engine.evaluateLikelihood()
                rows = []
                for call in range(args.warmup + args.calls):
                    point = center if scenario == 'constant' else values[call % len(values)]
                    parameter.setParameterValue(point)
                    start = time.perf_counter()
                    engine.evaluateLikelihood()
                    elapsed = time.perf_counter() - start
                    likelihood = engine.getLikelihoodInterface().getLastLikelihood()
                    if not math.isfinite(likelihood):
                        raise RuntimeError(f'Nonfinite likelihood at {scenario}/{trial}/{call}')
                    if call < args.warmup:
                        continue
                    timing = manager.getLastTimingSummary()
                    rows.append({'value': point, 'likelihood': likelihood,
                                 'seconds': {**{f: getattr(timing, f) for f in SECONDS},
                                             'evaluationSeconds': elapsed,
                                             'hostPreparationSeconds': manager.getLastPreparationSeconds()},
                                 'counts': {f: getattr(timing, f) for f in COUNTS}})
                    if len(rows) % 10 == 0:
                        progress(f'{scenario}: trial {trial + 1}/{args.trials}, {len(rows)}/{args.calls}')
                trials.append(rows)
                save(args.output / f'{scenario}-trial{trial + 1}.json', rows)
                all_rows = [row for group in trials for row in group]
                summary['scenarios'][scenario] = {
                    'completedTrials': len(trials),
                    'timings': {field: {
                        'medianMs': statistics.median(row['seconds'][field] for row in all_rows) * 1000,
                        'p95Ms': sorted(row['seconds'][field] for row in all_rows)[
                            int(.95 * (len(all_rows) - 1))] * 1000}
                        for field in rows[0]['seconds']}}
                save(args.output / 'summary.json', summary)
            # Copy outputs only outside the measured evaluations.
            manager.materialize('Histograms')
            samples = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
            save(args.output / f'{scenario}-histograms.json', [
                {'sample': sample.getName(), 'bins': [[v.sumWeights, v.sqrtSumSqWeights]
                 for v in sample.getHistogram().getBinContentList()]} for sample in samples])
    finally:
        parameter.setParameterValue(center)
    # Explicit host copies and Python event walks are kept outside all timings.
    # Record every point, rather than only the final histogram, for A/B checks.
    checks, previous = [], None
    for point in [center, *values]:
        parameter.setParameterValue(point)
        engine.evaluateLikelihood()
        manager.materialize('EventState')
        manager.materialize('Histograms')
        samples = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
        bins = array('i', (event.getIndices().bin for sample in samples for event in sample.getEventList()))
        changed = None if previous is None else sum(a != b for a, b in zip(previous, bins))
        checks.append({'value': point, 'events': len(bins), 'changedBins': changed,
                       'changedFraction': None if changed is None else changed / len(bins),
                       'binDigest': hashlib.sha256(bins.tobytes()).hexdigest(),
                       'likelihood': engine.getLikelihoodInterface().getLastLikelihood(),
                       'histograms': [{'sample': sample.getName(), 'bins': [
                           [v.sumWeights, v.sqrtSumSqWeights]
                           for v in sample.getHistogram().getBinContentList()]} for sample in samples]})
        previous = bins
    save(args.output / 'point-checks.json', checks)
    parameter.setParameterValue(center)
    summary['complete'] = True
    save(args.output / 'summary.json', summary)
    progress(f'Finished: {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--override', nargs='*', default=[])
    parser.add_argument('--work-dir', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path,
                        default=Path(f'benchmark-{Path(__file__).stem}-{time.strftime("%Y%m%d%H%M%S")}'),
                        help='Output directory (default: ./benchmark-<script-name>-YYYYMMDDHHmmss)')
    parser.add_argument('--parameter', default='pmuScale')
    parser.add_argument('--sigma-offsets', nargs='+', type=float, default=[-1., 1., -.5, .5])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--calls', type=int, default=60)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--trials', type=int, default=3)
    args = parser.parse_args()
    if min(args.threads, args.calls, args.trials) <= 0 or args.warmup < 0:
        parser.error('Counts must be positive; warmup may be zero')
    if len(args.sigma_offsets) < 2 or not all(math.isfinite(v) for v in args.sigma_offsets):
        parser.error('Provide at least two finite sigma offsets')
    args.work_dir = args.work_dir.resolve()
    args.config = (args.work_dir / args.config).resolve()
    args.output = args.output.resolve()
    if not args.config.is_file():
        parser.error(f'Missing configuration: {args.config}')
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'settings.json', {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'python': sys.executable})
    measure(args)


if __name__ == '__main__':
    main()
