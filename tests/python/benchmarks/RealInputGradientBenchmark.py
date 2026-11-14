#!/usr/bin/env python3
"""Benchmark arbitrary GUNDAM inputs, in isolated processes per backend/data mode.

Run directly with the desired GUNDAM module on PYTHONPATH. Initialization and
warmup are excluded from callback timings. HESSE and output plots are disabled.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-of', '--overrides', nargs='*', default=[])
    parser.add_argument('--backends', nargs='+', choices=['Original', 'CPU', 'MPS', 'CUDA'], default=['Original', 'CPU'])
    parser.add_argument('--data-modes', nargs='+', choices=['Asimov', 'Toy'], default=['Asimov', 'Toy'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--points', type=int, default=8)
    parser.add_argument('--amplitude', type=float, default=.01, help='Displacements in prior standard deviations')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--skip-fit', action='store_true')
    parser.add_argument('--max-fit-calls', type=int, default=30000, help='Fit evaluation budget; 0 keeps the input configuration')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.threads, args.points, args.repeats, args.trials) < 1 or args.amplitude < 0 or args.max_fit_calls < 0:
        parser.error('Counts must be positive; amplitude and fit budget must be nonnegative')
    args.output = args.output.resolve()
    args.work_dir = args.work_dir.resolve()
    if args.worker is None:
        args.output.mkdir(parents=True, exist_ok=False)
        results = []
        for mode in args.data_modes:
            for backend in args.backends:
                label = mode.lower() + '-' + backend.lower()
                command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker', mode, backend]
                print('Running', label, flush=True)
                with (args.output / (label + '.log')).open('w') as log:
                    status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
                path = args.output / (label + '.json')
                results.append({'case': label, 'exitCode': status, 'report': str(path)})
                (args.output / 'summary.json').write_text(json.dumps(results, indent=2))
                print(label, 'exit', status, flush=True)
        return int(any(x['exitCode'] for x in results))

    import GUNDAM as G
    mode, backend = args.worker
    label = mode.lower() + '-' + backend.lower()
    G.setRuntimeWorkingDirectory(str(args.work_dir))
    G.setNumberOfThreads(args.threads)
    G.setLightOutputMode(True)
    G.FitterEngine.setRandomSeed(args.seed)
    builder = G.ConfigUtils.ConfigBuilder()
    builder.setConfig(args.config)
    for override in args.overrides:
        builder.override(override)
    # Remove a legacy budget alias before overriding its canonical field.
    if args.max_fit_calls:
        config = json.loads(builder.toString())
        config['fitterEngineConfig']['minimizerConfig'].pop('max_fcn', None)
        builder.setConfig(G.JsonType(json.dumps(config)))
    # Keep input physics and minimizer settings; explicitly choose the measured path.
    builder.override(G.JsonType(json.dumps({'fitterEngineConfig': {
        'generateSamplePlots': False, 'generateOneSigmaPlots': False,
        'minimizerConfig': {'useAnalyticGradient': True, 'useNormalizedFitSpace': True,
                            'enablePostFitErrorFit': False,
                            **({'maxFcnCalls': args.max_fit_calls} if args.max_fit_calls else {})},
        'likelihoodInterfaceConfig': {'backendManagerConfig': {
            'isEnabled': backend != 'Original', 'useBackend': 'CPU' if backend == 'Original' else backend}}}})))
    (args.output / (label + '-config.json')).write_text(builder.toString())
    reader = G.ConfigUtils.ConfigReader(builder.getConfig())
    reader.defineField(G.ConfigUtils.ConfigReader.FieldDefinition('fitterEngineConfig'))
    engine = G.FitterEngine()
    engine.configure(reader.fetchValueConfigReader('fitterEngineConfig'))
    likelihood = engine.getLikelihoodInterface()
    likelihood.setDataType(G.LikelihoodInterface.DataType(mode))
    likelihood.setForceAsimovData(mode == 'Asimov')
    minimizer = engine.getMinimizer()
    minimizer.setDisableCalcError(True)
    started = time.perf_counter()
    engine.initialize()
    initialization = time.perf_counter() - started
    manager = engine.getBackendsManager()
    if backend != 'Original':
        if manager.getActiveBackendName() != backend:
            raise RuntimeError('Requested backend is not active')
        manager.setEnableAutoMaterialize(False)
    pars = list(minimizer.getMinimizerFitParameterList())
    # evalFit uses normalized coordinates because useNormalizedFitSpace is forced.
    scales = [p.getStepSize() if math.isnan(p.getStdDevValue()) else p.getStdDevValue() for p in pars]
    start = [(p.getParameterValue() - p.getPriorValue()) / scale for p, scale in zip(pars, scales)]
    points = [start]
    for j in range(1, args.points):
        point = []
        for i, p in enumerate(pars):
            delta = args.amplitude * math.sin(.7 * i + .43 * j)
            value = start[i] + delta
            point.append(value if p.isInDomain(p.getPriorValue() + value * scales[i]) else start[i])
        points.append(point)
    report = {'complete': False, 'module': G.__file__, 'platform': platform.platform(),
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'parameters': len(pars), 'events': likelihood.getModelPropagator().getSampleSet().getNbOfEvents(),
              'initializationSeconds': initialization, 'points': points, 'runs': [],
              'dataHistograms': {s.getName(): [[b.sumWeights, b.sqrtSumSqWeights]
                  for b in s.getHistogram().getBinContentList()]
                  for s in likelihood.getDataPropagator().getSampleSet().getSampleList()}}
    path = args.output / (label + '.json')
    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False))
    save()
    for point in points:
        minimizer.evalFit(point)
        minimizer.evalFitGradientAt(point)
    timing_fields = ['cachePreparationSeconds', 'cachedDialStageSeconds', 'eventWeightsStageSeconds',
                     'histogramStageSeconds', 'likelihoodHostSeconds']
    for trial in range(args.trials):
        run = {'trial': trial}
        for kind in (['value', 'gradient'] if trial % 2 == 0 else ['gradient', 'value']):
            callback = minimizer.evalFit if kind == 'value' else minimizer.evalFitGradientAt
            checksum = 0.
            started = time.perf_counter()
            for _ in range(args.repeats):
                for point in points:
                    value = callback(point)
                    checksum += value if kind == 'value' else sum(value)
            count = args.repeats * len(points)
            run[kind + 'SecondsPerCall'] = (time.perf_counter() - started) / count
            run[kind + 'Checksum'] = checksum
            # Separate untimed pass: backend stage clocks describe propagation only,
            # not the derivative stages. Average over the same parameter points.
            if backend != 'Original':
                stages = dict.fromkeys(timing_fields, 0.)
                for point in points:
                    callback(point)
                    timing = manager.getLastTimingSummary()
                    for field in stages:
                        stages[field] += getattr(timing, field) / len(points)
                run[kind + 'PropagationStagesSeconds'] = stages
        report['runs'].append(run)
        save()
        print('TRIAL', json.dumps(run), flush=True)
    report['medianSecondsPerCall'] = {kind: statistics.median(r[kind + 'SecondsPerCall'] for r in report['runs'])
                                      for kind in ['value', 'gradient']}
    save()
    # Restore the original start before minimizing; benchmark calls move parameters.
    minimizer.evalFit(start)
    minimizer.updateFitParameterSettings()
    if not args.skip_fit:
        started = time.perf_counter()
        minimizer.minimize()
        report['fit'] = {'seconds': time.perf_counter() - started,
                         'converged': minimizer.hasConverged(), 'edm': minimizer.getEdm(),
                         'valueCalls': minimizer.getNbFunctionCalls(), 'gradientCalls': minimizer.getNbGradientCalls(),
                         'likelihood': likelihood.getLastLikelihood(),
                         'statistical': likelihood.getLastStatLikelihood(), 'penalty': likelihood.getLastPenaltyLikelihood(),
                         'parameters': [p.getParameterValue() for p in pars]}
    report['complete'] = True
    save()
    print('RESULT', path, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
