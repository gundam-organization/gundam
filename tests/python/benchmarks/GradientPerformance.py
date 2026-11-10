#!/usr/bin/env python3
"""Compare original-engine and CPU gradient callbacks on existing GUNDAM inputs.

Accepts gundamFitter's -c/-of/-t/-a. See GradientPerformance.md for measurement
scope, native stack profiling and the output format. No minimization is run.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stats(values):
    return {'median': statistics.median(values), 'mean': statistics.mean(values),
            'min': min(values), 'max': max(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.}


def profile_native(args, directory, evaluate, points):
    """Separate resident-gradient pass; initialization and timing trials excluded."""
    if args.profile == 'none':
        return {'enabled': False}
    sampler = shutil.which('sample') if sys.platform == 'darwin' else None
    perf = shutil.which('perf') if sys.platform.startswith('linux') else None
    if not sampler and not perf:
        return {'enabled': False, 'reason': 'Neither macOS sample nor Linux perf available'}
    status = {'enabled': True, 'complete': False, 'calls': 0,
              'scope': 'gradient callbacks only, all native threads; includes Python conversion'}
    if sampler:
        command = [sampler, str(os.getpid()), str(args.profile_seconds), '1',
                   '-fullPaths', '-file', str(directory / 'gradient-sample.txt')]
    else:
        command = [perf, 'record', '-e', 'cycles:u', '-F', '199', '--call-graph', 'dwarf,8192',
                   '-p', str(os.getpid()), '-o', str(directory / 'gradient-perf.data')]
    status['command'] = command
    write_json(directory / 'profile.json', status)
    with (directory / 'profile.log').open('w') as log:
        try:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        except OSError as error:
            status.update(enabled=False, reason=str(error))
            write_json(directory / 'profile.json', status)
            return status
        started = time.monotonic()
        # A bounded loop also protects against a profiler that fails to stop.
        deadline = started + args.profile_seconds + 3.
        try:
            while child.poll() is None and time.monotonic() < deadline:
                evaluate(points[status['calls'] % len(points)])
                status['calls'] += 1
                if perf and time.monotonic() - started >= args.profile_seconds:
                    break
        finally:
            stopped_perf = perf and child.poll() is None
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            status['exitCode'] = child.returncode
            status['wallSeconds'] = time.monotonic() - started
            status['complete'] = bool(status['calls'] and (child.returncode == 0 or
                (stopped_perf and child.returncode == -signal.SIGINT)))
    if perf and status['complete']:
        with (directory / 'gradient-perf-report.txt').open('w') as output:
            try:
                result = subprocess.run([perf, 'report', '--stdio', '--no-children',
                    '--sort', 'dso,symbol', '--percent-limit', '.5', '-g', 'none',
                    '-i', str(directory / 'gradient-perf.data')],
                    stdout=output, stderr=subprocess.STDOUT, timeout=30)
                status['reportExitCode'] = result.returncode
            except (OSError, subprocess.TimeoutExpired) as error:
                status['reportError'] = str(error)
    write_json(directory / 'profile.json', status)
    return status


def worker(args):
    if args.python_module_dir:
        sys.path.insert(0, str(args.python_module_dir))
    import GUNDAM as G
    directory = args.output / args.worker.lower()
    directory.mkdir()
    report = {'complete': False, 'backend': args.worker, 'module': G.__file__,
              'platform': platform.platform(), 'python': sys.version, 'pid': os.getpid(),
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'runs': [], 'checks': []}
    path = directory / 'report.json'
    write_json(path, report)
    try:
        G.setRuntimeWorkingDirectory(str(args.work_dir))
        G.setNumberOfThreads(args.threads)
        G.setLightOutputMode(True)
        G.FitterEngine.setRandomSeed(args.seed)
        builder = G.ConfigUtils.ConfigBuilder()
        builder.setConfig(args.config)
        for override in args.overrides:
            builder.override(override)
        # Preserve physics and coordinate choices; override only execution path
        # and disable output work that is irrelevant to callback measurements.
        builder.override(G.JsonType(json.dumps({'fitterEngineConfig': {
            'generateSamplePlots': False, 'generateOneSigmaPlots': False,
            'minimizerConfig': {'useAnalyticGradient': True, 'writeLlhHistory': False,
                                'enablePostFitErrorFit': False},
            'likelihoodInterfaceConfig': {'backendManagerConfig': {
                'isEnabled': args.worker == 'CPU', 'useBackend': 'CPU'}}}})))
        resolved = json.loads(builder.toString())
        (directory / 'config.json').write_text(builder.toString())
        config = resolved['fitterEngineConfig']
        normalized = config.get('minimizerConfig', {}).get('useNormalizedFitSpace', True)
        reader = G.ConfigUtils.ConfigReader(builder.getConfig())
        reader.defineField(G.ConfigUtils.ConfigReader.FieldDefinition('fitterEngineConfig'))
        engine = G.FitterEngine()
        engine.configure(reader.fetchValueConfigReader('fitterEngineConfig'))
        likelihood = engine.getLikelihoodInterface()
        likelihood.setDataType(G.LikelihoodInterface.DataType.Asimov if args.asimov
                               else G.LikelihoodInterface.DataType.RealData)
        likelihood.setForceAsimovData(args.asimov)
        minimizer = engine.getMinimizer()
        minimizer.setDisableCalcError(True)
        started = time.perf_counter()
        engine.initialize()
        report['initializationSeconds'] = time.perf_counter() - started
        manager = engine.getBackendsManager()
        if args.worker == 'CPU':
            if manager.getActiveBackendName() != 'CPU':
                raise RuntimeError('Requested CPU backend is not active')
            manager.setEnableAutoMaterialize(False)
        elif manager.hasBackend():
            raise RuntimeError('Original-engine case unexpectedly has an active backend')
        parameters = list(minimizer.getMinimizerFitParameterList())
        if not parameters:
            raise RuntimeError('No fit parameters to benchmark')
        scales = [p.getStdDevValue() if math.isfinite(p.getStdDevValue()) and p.getStdDevValue() > 0.
                  else p.getStepSize() for p in parameters]
        if any(not math.isfinite(s) or s <= 0. for s in scales):
            raise RuntimeError('Cannot construct points: invalid parameter scale/step')
        center = [(p.getParameterValue()-p.getPriorValue())/s if normalized else p.getParameterValue()
                  for p, s in zip(parameters, scales)]
        points = [center]
        for j in range(1, args.points):
            point = []
            for i, (p, s) in enumerate(zip(parameters, scales)):
                delta = args.amplitude * math.sin(.7*i+.43*j)
                value = center[i] + delta*(1. if normalized else s)
                physical = p.getPriorValue()+s*value if normalized else value
                point.append(value if p.isInDomain(physical) else center[i])
            points.append(point)
        report.update(normalizedFitSpace=normalized, points=points,
            parameterNames=[p.getFullTitle() for p in parameters],
            events=likelihood.getModelPropagator().getSampleSet().getNbOfEvents(),
            dataHistograms={s.getName(): [[b.sumWeights, b.sqrtSumSqWeights]
                for b in s.getHistogram().getBinContentList()]
                for s in likelihood.getDataPropagator().getSampleSet().getSampleList()})
        module = Path(G.__file__).resolve()
        for parent in module.parents:
            cache = parent / 'CMakeCache.txt'
            if cache.exists():
                report['buildCache'] = str(cache)
                report['buildOptions'] = [line for line in cache.read_text().splitlines()
                    if line.startswith(('CMAKE_BUILD_TYPE:', 'CMAKE_CXX_COMPILER:',
                                        'CMAKE_CXX_FLAGS', 'WITH_', 'USE_STATIC_LINKS:'))]
                break
        write_json(path, report)
        # Verification and warmup are outside timing loops.
        for point in points:
            value = minimizer.evalFit(point)
            gradient = list(minimizer.evalFitGradientAt(point))
            if not math.isfinite(value) or not all(math.isfinite(x) for x in gradient):
                raise RuntimeError('Non-finite likelihood/gradient at benchmark point; reduce --amplitude')
            report['checks'].append({'value': value, 'gradient': gradient})
        for _ in range(args.warmup):
            for point in points:
                minimizer.evalFitGradientAt(point)
        write_json(path, report)
        timing_fields = ['propagationWallSeconds', 'cachePreparationSeconds', 'cachedDialStageSeconds',
            'variableStageSeconds', 'binningStageSeconds', 'eventWeightsStageSeconds',
            'histogramStageSeconds', 'likelihoodHostSeconds']
        for workload, sequence in [('varying', points), ('unchanged', [points[-1]])]:
            for trial in range(args.trials):
                run = {'workload': workload, 'trial': trial, 'calls': args.repeats*len(sequence)}
                for kind in (['value', 'gradient'] if trial % 2 == 0 else ['gradient', 'value']):
                    evaluate = minimizer.evalFit if kind == 'value' else minimizer.evalFitGradientAt
                    # Settle the cache before each measurement, especially unchanged inputs.
                    evaluate(sequence[-1])
                    cpu_start = time.process_time()
                    started = time.perf_counter()
                    for _ in range(args.repeats):
                        for point in sequence:
                            result = evaluate(point)
                    wall = time.perf_counter()-started
                    cpu = time.process_time()-cpu_start
                    run[kind+'WallSecondsPerCall'] = wall/run['calls']
                    run[kind+'CpuSecondsPerCall'] = cpu/run['calls']
                    run[kind+'LastChecksum'] = float(result) if kind == 'value' else sum(result)
                    if args.worker == 'CPU':
                        # Untimed pass: stage clocks are propagation measurements.
                        stages = {field: [] for field in timing_fields}
                        for point in sequence:
                            evaluate(point)
                            timing = manager.getLastTimingSummary()
                            for field in stages:
                                stages[field].append(getattr(timing, field))
                        run[kind+'PropagationStagesSeconds'] = {k: statistics.mean(v) for k, v in stages.items()}
                report['runs'].append(run)
                write_json(path, report)
                print('TRIAL', json.dumps(run), flush=True)
        report['timings'] = {workload: {kind: stats([r[kind+'WallSecondsPerCall'] for r in report['runs']
            if r['workload'] == workload]) for kind in ('value', 'gradient')}
            for workload in ('varying', 'unchanged')}
        write_json(path, report)
        # Sampling follows all primary timings and runs only varying gradients.
        report['profile'] = profile_native(args, directory, minimizer.evalFitGradientAt, points)
        report['complete'] = True
        write_json(path, report)
    except Exception:
        report['error'] = traceback.format_exc()
        write_json(path, report)
        raise
    return 0


def compare(reports, rtol, atol):
    if any(not r.get('complete') for r in reports.values()) or len(reports) != 2:
        return {'passed': None, 'reason': 'Not run: both complete reports are required'}
    native, cpu = reports['Original'], reports['CPU']
    if any(native[k] != cpu[k] for k in ('parameterNames', 'points', 'dataHistograms', 'events', 'normalizedFitSpace')):
        return {'passed': False, 'reason': 'The two workloads differ'}
    checks = []
    for i, (a, b) in enumerate(zip(native['checks'], cpu['checks'])):
        if len(a['gradient']) != len(b['gradient']):
            return {'passed': False, 'reason': 'Gradient dimensions differ'}
        differences = [abs(x-y) for x, y in zip(a['gradient'], b['gradient'])]
        checks.append({'point': i, 'valueMatches': math.isclose(a['value'], b['value'], rel_tol=rtol, abs_tol=atol),
            'gradientMatches': all(math.isclose(x, y, rel_tol=rtol, abs_tol=atol)
                                   for x, y in zip(a['gradient'], b['gradient'])),
            'maxAbsoluteGradientDifference': max(differences, default=0.)})
    return {'passed': all(c['valueMatches'] and c['gradientMatches'] for c in checks),
            'rtol': rtol, 'atol': atol, 'points': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-of', '--override-files', '--overrides', dest='overrides', nargs='+', action='extend', default=[])
    parser.add_argument('-t', '--threads', type=int, default=4)
    parser.add_argument('-a', '--asimov', action='store_true')
    parser.add_argument('--work-dir', type=Path, default=Path.cwd())
    parser.add_argument('--python-module-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--points', type=int, default=8)
    parser.add_argument('--amplitude', type=float, default=.01, help='Perturbation in parameter standard deviations')
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--profile', choices=['auto', 'none'], default='auto')
    parser.add_argument('--profile-seconds', type=int, default=10)
    parser.add_argument('--rtol', type=float, default=3e-5)
    parser.add_argument('--atol', type=float, default=3e-6)
    parser.add_argument('--worker', choices=['Original', 'CPU'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.threads, args.points, args.trials, args.repeats, args.profile_seconds) < 1 or args.warmup < 0:
        parser.error('Counts must be positive, warmup nonnegative')
    if any(not math.isfinite(x) or x < 0. for x in (args.amplitude, args.rtol, args.atol)):
        parser.error('Amplitude and comparison tolerances must be finite and nonnegative')
    for name in ('work_dir', 'output', 'python_module_dir'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(['git', '-C', str(Path(__file__).resolve().parents[3]),
        'describe', '--always', '--dirty'], text=True, capture_output=True)
    summary = {'complete': False, 'sourceRevision': revision.stdout.strip(), 'cases': {}}
    reports = {}
    for backend in ('Original', 'CPU'):
        print(f'{backend}: loading inputs, measuring, then profiling; log in {args.output}', flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker', backend]
        with (args.output / (backend.lower()+'.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        report_path = args.output / backend.lower() / 'report.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {'complete': False}
        reports[backend] = report
        summary['cases'][backend] = {'exitCode': result.returncode, 'report': str(report_path),
            'timings': report.get('timings'), 'profile': report.get('profile'),
            'error': report.get('error')}
        write_json(args.output / 'summary.json', summary)
        print(f'{backend}: exit {result.returncode}', flush=True)
        if result.returncode:
            error = report.get('error', '').strip().splitlines()
            reason = error[-1] if error else f'See {args.output / (backend.lower()+".log")}'
            print(f'{backend}: {reason}', flush=True)
        profile = report.get('profile')
        if args.profile != 'none' and profile is not None and not profile.get('complete'):
            print(f'{backend}: native profile unavailable/incomplete; see profile.json and profile.log', flush=True)
    summary['comparison'] = compare(reports, args.rtol, args.atol)
    if all(r.get('timings') for r in reports.values()):
        summary['cpuSpeedup'] = {w: reports['Original']['timings'][w]['gradient']['median'] /
            reports['CPU']['timings'][w]['gradient']['median'] for w in ('varying', 'unchanged')}
        for w, speedup in summary['cpuSpeedup'].items():
            print(f'{w}: Original {1e3*reports["Original"]["timings"][w]["gradient"]["median"]:.3f} ms, '
                  f'CPU {1e3*reports["CPU"]["timings"][w]["gradient"]["median"]:.3f} ms, '
                  f'speedup {speedup:.3f}x', flush=True)
    summary['complete'] = all(r.get('complete') for r in reports.values())
    write_json(args.output / 'summary.json', summary)
    comparison = summary['comparison']['passed']
    print('Gradient/value comparison:', 'NOT RUN' if comparison is None else ('PASS' if comparison else 'FAIL'), flush=True)
    print('Report:', args.output / 'summary.json', flush=True)
    return int(not summary['complete'] or not summary['comparison']['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
