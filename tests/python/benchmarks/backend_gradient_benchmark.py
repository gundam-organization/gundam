"""Shared CPU/Metal analytic-gradient microbenchmark (no minimization).

Timings include propagation, likelihood, penalty and Python callback overhead.
Returning the gradient to Python also waits for device results on MPS.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time


def load_fixture():
    spec = importlib.util.spec_from_file_location(
        'gradient_fixture', Path(__file__).with_name('700BackendsCommonGradientBenchmark.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(backend):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=int, default=120000)
    parser.add_argument('--bins', type=int, default=48)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--points', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--check-components', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path.cwd() / (backend.lower() + '-gradient'))
    args = parser.parse_args()
    if min(args.events, args.bins, args.threads, args.points, args.repeats,
           args.trials, args.warmup, args.check_components) <= 0 or args.events < args.bins:
        parser.error('Counts must be positive, with at least one event per bin')
    if backend == 'MPS' and platform.system() != 'Darwin':
        print('SKIP: MPS requires macOS/Metal')
        return 0

    import GUNDAM
    fixture = load_fixture()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f'Results: {output}', flush=True)
    input_path = output / 'input.root'
    fixture.write_input(input_path, args.events, args.bins)
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(args.threads)
    config = fixture.configuration(input_path, args.bins, True, 0)
    config['likelihoodInterfaceConfig']['backendManagerConfig'] = {
        'isEnabled': True, 'useBackend': backend}
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    engine.initialize()
    manager = engine.getBackendsManager()
    if manager.getActiveBackendName() != backend:
        raise AssertionError(f'Requested {backend}, active backend: {manager.getActiveBackendName()}')
    manager.setEnableAutoMaterialize(False)
    minimizer = engine.getMinimizer()
    dimension = len(minimizer.getMinimizerFitParameterList())
    if dimension != args.bins + 2:
        raise AssertionError(f'Unexpected fit dimension: {dimension}')
    points = [[.2 * math.sin(.7 * i + .43 * j + .4) for i in range(dimension)]
              for j in range(args.points)]

    # Validate outside timers, including both dense Graph parameters and sparse
    # per-bin normalizations. MPS differences need a larger step for FP32 weights.
    step = 2e-3 if backend == 'MPS' else 2e-5
    rel_tol, abs_tol = (5e-3, 5e-2) if backend == 'MPS' else (3e-5, 3e-6)
    count = min(args.check_components, dimension)
    indices = sorted({dimension - 2, dimension - 1} | {
        round(i * (dimension - 1) / max(1, count - 1)) for i in range(count)})
    checks = []
    for point in (points[0], points[-1]):
        gradient = list(minimizer.evalFitGradientAt(point))
        if len(gradient) != dimension or not all(map(math.isfinite, gradient)):
            raise AssertionError('Invalid analytic gradient')
        for index in indices:
            up, down = point.copy(), point.copy()
            up[index] += step
            down[index] -= step
            numeric = (minimizer.evalFit(up) - minimizer.evalFit(down)) / (2 * step)
            actual = gradient[index]
            if not math.isfinite(numeric) or not math.isclose(actual, numeric, rel_tol=rel_tol, abs_tol=abs_tol):
                raise AssertionError(f'{backend} component {index}: analytic={actual}, numeric={numeric}')
            checks.append({'component': index, 'analytic': actual, 'numeric': numeric})

    for _ in range(args.warmup):
        for point in points:
            minimizer.evalFit(point)
            minimizer.evalFitGradientAt(point)

    report = {'complete': False, 'backend': backend, 'module': GUNDAM.__file__,
              'platform': platform.platform(), 'python': sys.version,
              'arguments': {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
              'parameters': dimension, 'finiteDifferenceChecks': checks,
              'finiteDifferenceStep': step, 'relativeTolerance': rel_tol, 'absoluteTolerance': abs_tol,
              'timingScope': 'Python callback, propagation, likelihood and penalty; no initialization or fit',
              'runs': []}

    def save():
        (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))

    save()
    for trial in range(args.trials):
        run = {'trial': trial}
        # Alternate order to reduce systematic thermal/order bias.
        for mode in (('value', 'gradient') if trial % 2 == 0 else ('gradient', 'value')):
            callback = minimizer.evalFit if mode == 'value' else minimizer.evalFitGradientAt
            checksum = 0.
            started = time.perf_counter()
            for _ in range(args.repeats):
                for point in points:
                    result = callback(point)
                    checksum += result if mode == 'value' else sum(result)
            elapsed = time.perf_counter() - started
            if not math.isfinite(checksum):
                raise AssertionError(f'Non-finite {mode} result')
            run[mode + 'SecondsPerCall'] = elapsed / (args.points * args.repeats)
            run[mode + 'Checksum'] = checksum
        report['runs'].append(run)
        save()
        print('RESULT', json.dumps(run), flush=True)
    report['medianSecondsPerCall'] = {
        mode: statistics.median(run[mode + 'SecondsPerCall'] for run in report['runs'])
        for mode in ('value', 'gradient')}
    report['gradientToValueRatio'] = (report['medianSecondsPerCall']['gradient'] /
                                      report['medianSecondsPerCall']['value'])
    report['complete'] = True
    save()
    print('MEDIAN', json.dumps(report['medianSecondsPerCall']), flush=True)
    return 0
