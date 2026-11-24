#!/usr/bin/env python3
"""Measure the original CPU likelihood evaluation with fixed event binning."""

import argparse
from datetime import datetime
from itertools import cycle
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time


# The original propagation timers retain the last ten calls.
CALLS_PER_BLOCK = 10


def write_json(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def profile_scenarios(output, points, evaluate, blocks):
    from CpuSamplingProfile import check_profile_support, record_profile

    profile_dir = output / 'profiles'
    profile_dir.mkdir()
    preflight = check_profile_support(profile_dir, write_json)
    if not preflight['available']:
        print(preflight['reason'], flush=True)
        return

    for name, sequence in points.items():
        scenario_dir = profile_dir / name
        scenario_dir.mkdir()
        for point in sequence:
            evaluate(point)
        replay = cycle(sequence)
        print(f'Profiling {name}', flush=True)
        record_profile(scenario_dir, lambda: evaluate(next(replay)), blocks * CALLS_PER_BLOCK,
                       write_json, lambda message: print(message, flush=True), include_call_graph=True)


def make_points(parameters, normalized, amplitude, seed):
    center = [parameter.getParameterValue() for parameter in parameters]

    def fit_coordinates(values):
        if not normalized:
            return values
        return [parameter.toNormalizedValue(value) for parameter, value in zip(parameters, values)]

    def shifted(index, fraction):
        parameter = parameters[index]
        scale = parameter.getStdDevValue()
        if not math.isfinite(scale) or scale <= 0:
            scale = parameter.getStepSize()
        value = center[index] + fraction * scale
        if math.isfinite(value) and parameter.isInDomain(value) and parameter.isValidValue(value):
            return value
        return center[index]

    nominal = fit_coordinates(center)
    if not all(map(math.isfinite, nominal)):
        raise RuntimeError('The initial minimizer coordinates must be finite.')
    movable = [i for i, value in enumerate(center)
               if shifted(i, amplitude) != value or shifted(i, -amplitude) != value]
    if not movable:
        raise RuntimeError('No fit parameter can move with the requested amplitude.')

    rng = random.Random(seed)
    selected = rng.sample(movable, min(len(movable), CALLS_PER_BLOCK // 2))
    single = []
    for j in range(CALLS_PER_BLOCK // 2):
        i = selected[j % len(selected)]
        point = center.copy()
        point[i] = shifted(i, amplitude)
        if point[i] == center[i]:
            point[i] = shifted(i, -amplitude)
        single.extend((nominal, fit_coordinates(point)))

    multiple = []
    for j in range(CALLS_PER_BLOCK):
        point = [shifted(i, amplitude * math.sin(1.1 * j + 0.7 * i)) for i in range(len(center))]
        multiple.append(fit_coordinates(point))
    for points in (single, multiple):
        if any(a == b for a, b in zip(points, points[1:] + points[:1])):
            raise RuntimeError('A benchmark point repeats the previous point. Try a different amplitude.')
    return {'singleParameter': single, 'multipleParameters': multiple}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=Path.cwd())
    parser.add_argument('-c', '--config', default='configOA2024.yaml')
    parser.add_argument('-of', '--override', nargs='+', default=[])
    parser.add_argument('-t', '--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--amplitude', type=float, default=0.01,
                        help='Maximum shift in units of prior sigma (or nominal step size for flat parameters)')
    parser.add_argument('--blocks', type=int, default=20, help='Measured blocks of ten calls per scenario')
    parser.add_argument('--profile', action='store_true', help='Record a separate Linux perf pass per scenario')
    parser.add_argument('--profile-blocks', type=int, default=3, help='Profiled blocks of ten calls per scenario')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.threads, args.blocks, args.profile_blocks) < 1 or not 1 <= args.seed < 2**32:
        parser.error('Threads and blocks must be positive, and seed must be in [1, 2**32).')
    if not math.isfinite(args.amplitude) or args.amplitude <= 0:
        parser.error('Amplitude must be positive and finite.')

    import GUNDAM as G

    if not hasattr(G, 'disableLegacyCacheManager'):
        parser.error('Select a GUNDAM Python module built with the original CPU benchmark bindings.')

    work_dir = args.work_dir.resolve()
    default_output = 'benchmark-OriginalReweightBenchmark-' + datetime.now().strftime('%Y%m%d%H%M%S')
    output = (args.output or Path(default_output)).resolve()
    module = str(Path(G.__file__).resolve())
    G.setRuntimeWorkingDirectory(str(work_dir))
    G.setNumberOfThreads(args.threads)
    G.setLightOutputMode(True)
    G.disableLegacyCacheManager()
    G.FitterEngine.setRandomSeed(args.seed)

    builder = G.ConfigUtils.ConfigBuilder()
    builder.setConfig(args.config)
    builder.override(args.override)
    engine = G.FitterEngine()
    has_backends = hasattr(engine, 'getBackendsManager')
    overrides = {'generateSamplePlots': False, 'generateOneSigmaPlots': False}
    if has_backends:
        overrides['likelihoodInterfaceConfig'] = {'backendManagerConfig': {'isEnabled': False}}
    builder.override(G.JsonType(json.dumps({'fitterEngineConfig': overrides})))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(builder.toString())

    reader = G.ConfigUtils.ConfigReader(builder.getConfig())
    reader.defineField(G.ConfigUtils.ConfigReader.FieldDefinition('fitterEngineConfig'))
    engine.configure(reader.fetchValueConfigReader('fitterEngineConfig'))
    likelihood = engine.getLikelihoodInterface()
    likelihood.setDataType(G.LikelihoodInterface.DataType.RealData)
    minimizer = engine.getMinimizer()
    minimizer.setDisableCalcError(True)
    started = time.perf_counter()
    engine.initialize()
    initialization = time.perf_counter() - started
    if has_backends and engine.getBackendsManager().getActiveBackendName():
        raise RuntimeError('The benchmark requires the original CPU path.')

    model = likelihood.getModelPropagator()
    samples = model.getSampleSet()
    if (any(sample.isUseDynamicBinning() or sample.getHistogram().isBinningResponseEnabled()
            for sample in samples.getSampleList()) or model.getEventDialCache().hasVariableResponses()):
        raise RuntimeError('The benchmark requires fixed event binning and weight-only dials.')

    parameters = list(minimizer.getMinimizerFitParameterList())
    normalized = minimizer.useNormalizedFitSpace()
    points = make_points(parameters, normalized, args.amplitude, args.seed)
    evaluate = minimizer.evalFit
    report = {
        'engine': 'Original',
        'dataMode': 'RealData',
        'module': module,
        'gundamVersion': G.__version__,
        'platform': platform.platform(),
        'host': platform.node(),
        'workDir': str(work_dir),
        'threads': args.threads,
        'seed': args.seed,
        'amplitude': args.amplitude,
        'initializationSeconds': initialization,
        'modelEvents': samples.getNbOfEvents(),
        'normalizedFitSpace': normalized,
        'parameters': [parameter.getFullTitle() for parameter in parameters],
        'scenarios': {},
    }

    for name, sequence in points.items():
        reference = []
        for point in sequence:
            value = evaluate(point)
            components = {
                'total': value,
                'statistical': likelihood.getLastStatLikelihood(),
                'penalty': likelihood.getLastPenaltyLikelihood(),
            }
            if not all(map(math.isfinite, components.values())):
                raise RuntimeError(f'{name} point {len(reference)} has a non-finite likelihood.')
            reference.append(components)

        blocks = []
        for _ in range(args.blocks):
            started = time.perf_counter()
            values = [evaluate(point) for point in sequence]
            elapsed = time.perf_counter() - started
            if not all(math.isclose(value, expected['total'], rel_tol=1e-9, abs_tol=1e-9)
                       for value, expected in zip(values, reference)):
                raise RuntimeError(f'{name} likelihood changed between repeated evaluations.')
            blocks.append({'callbackSeconds': elapsed / CALLS_PER_BLOCK,
                           **model.getCpuPropagationTimings()})

        median = {stage: statistics.median(block[stage] for block in blocks) for stage in blocks[0]}
        report['scenarios'][name] = {
            'points': sequence,
            'likelihoods': reference,
            'callsPerBlock': CALLS_PER_BLOCK,
            'meanSecondsPerCallByBlock': blocks,
            'medianSecondsPerCall': median,
        }
        print(name, ', '.join(f'{stage}={seconds * 1000:.3f} ms' for stage, seconds in median.items()),
              flush=True)

    path = output / 'summary.json'
    write_json(path, report)
    print(path, flush=True)
    if args.profile:
        profile_scenarios(output, points, evaluate, args.profile_blocks)


if __name__ == '__main__':
    main()
