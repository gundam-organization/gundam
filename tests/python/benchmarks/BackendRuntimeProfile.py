#!/usr/bin/env python3
"""Break down real-input propagation and analytic gradients with one dataset load.

Accepts gundamFitter's -c/-of/-t/-a. No minimization is performed.
See BackendRuntimeProfile.md for timer boundaries and interpretation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback


GRADIENT_FIELDS = (
    'gradientWallSeconds', 'gradientModelPreparationSeconds', 'gradientDeviceSeconds',
    'gradientProducerUploadSeconds', 'gradientDialStageSeconds', 'gradientBinStageSeconds',
    'gradientEventStageSeconds', 'gradientReductionStageSeconds', 'gradientReadbackSeconds')
PROPAGATION_FIELDS = (
    'propagationWallSeconds', 'cachePreparationSeconds', 'cacheExecutionSeconds',
    'eventSelectionSeconds', 'parameterUploadSeconds', 'cachedDialStageSeconds',
    'eventWeightsStageSeconds', 'variableStageSeconds', 'binningStageSeconds',
    'histogramStageSeconds', 'likelihoodDeviceSeconds', 'likelihoodHostSeconds',
    'devicePropagationSeconds', 'externalWeightUploadSeconds', 'histogramReadbackSeconds',
    'eventWeightReadbackSeconds', 'eventStateReadbackSeconds')
COUNT_FIELDS = (
    'uniqueDialCount', 'eventDialIndexCount', 'cacheCandidateCount', 'cacheInputTransformCount',
    'eventWeightCandidateCount', 'eventWeightsRecomputedCount', 'eventWeightCoverageKnown',
    'eventSelectionCacheHit', 'sparseEventWeightsUsed', 'rebinnedEventCount', 'binCandidateCount',
    'binContributionCount', 'contributionBufferBytes', 'externalWeightUploadBytes',
    'histogramReadbackBytes', 'eventWeightReadbackBytes', 'eventStateReadbackBytes',
    'likelihoodReadbackBytes')
GPU_PROPAGATION = ('parameterUploadSeconds', 'cachedDialStageSeconds', 'eventWeightsStageSeconds',
                   'variableStageSeconds', 'binningStageSeconds', 'histogramStageSeconds',
                   'likelihoodDeviceSeconds')
GPU_GRADIENT = ('gradientProducerUploadSeconds', 'gradientDialStageSeconds', 'gradientBinStageSeconds',
                'gradientEventStageSeconds', 'gradientReductionStageSeconds', 'gradientReadbackSeconds')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def progress(message):
    print(f'[benchmark] {message}', flush=True)


def distribution(values):
    ordered = sorted(values)
    return dict(mean=statistics.mean(values), median=statistics.median(values), min=min(values),
                max=max(values), p95=ordered[math.ceil(.95 * len(ordered)) - 1], count=len(values))


def snapshot(manager, include_gradient):
    timing = manager.getLastTimingSummary()
    fields = PROPAGATION_FIELDS + COUNT_FIELDS + (GRADIENT_FIELDS if include_gradient else ())
    result = {name: getattr(timing, name) for name in fields}
    result['managerPreparationSeconds'] = manager.getLastPreparationSeconds()
    return result


def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return dict(command=command, exitCode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=command, error=str(error))


def coordinate_manifest(parameters, normalized, amplitude):
    center, manifest, alternatives = [], [], []
    for parameter in parameters:
        physical = parameter.getParameterValue()
        sigma = parameter.getStdDevValue()
        if math.isnan(sigma):
            sigma = parameter.getStepSize()
        if not math.isfinite(sigma) or sigma <= 0.:
            raise RuntimeError('Invalid proposal scale: ' + parameter.getFullTitle())
        value = (physical - parameter.getPriorValue()) / sigma if normalized else physical
        center.append(value)
        proposals = []
        for sign in (-1., 1.):
            step = amplitude * sigma * sign
            for _ in range(60):
                candidate = physical + step
                if candidate != physical and math.isfinite(candidate) and parameter.isInDomain(candidate):
                    proposals.append(value + step / sigma if normalized else candidate)
                    break
                step *= .5
        alternatives.append(proposals)
        manifest.append(dict(name=parameter.getFullTitle(), center=value, physicalCenter=physical,
                             sigma=sigma, proposals=proposals))
    return center, manifest, alternatives


def summarize(rows, cuda=True):
    result = {}
    for scenario in dict.fromkeys(row['scenario'] for row in rows):
        result[scenario] = {}
        for workload in dict.fromkeys(row['workload'] for row in rows):
            selected = [row for row in rows if row['scenario'] == scenario and row['workload'] == workload]
            if not selected:
                continue
            entry = {'wallSeconds': distribution([row['wallSeconds'] for row in selected]),
                     'cpuSeconds': distribution([row['cpuSeconds'] for row in selected]), 'callbacks': {}}
            for callback in selected[0]['callbacks']:
                samples = [row['callbacks'][callback] for row in selected]
                timing = {name: distribution([sample[name] for sample in samples]) for name in samples[0]}
                ranking = sorted(GPU_PROPAGATION + (GPU_GRADIENT if callback == 'gradient' else ()),
                                 key=lambda field: timing[field]['median'], reverse=True) if cuda else []
                entry['callbacks'][callback] = dict(metrics=timing, gpuStageRanking=list(ranking))
            result[scenario][workload] = entry
    return result


def write_report(output, summary):
    lines = ['# Runtime profile', '',
             'One loaded engine. Durations below are medians in milliseconds.',
             'Initialization, lazy gradient setup, warmup and local-coordinate resets are excluded.', '',
             '| Scenario | Workload | Callback wall | Process CPU |', '|---|---|---:|---:|']
    for scenario, workloads in summary.items():
        for workload, entry in workloads.items():
            lines.append(f"| {scenario} | {workload} | {1000*entry['wallSeconds']['median']:.3f} | "
                         f"{1000*entry['cpuSeconds']['median']:.3f} |")
    for scenario, workloads in summary.items():
        for workload, entry in workloads.items():
            lines.extend(['', f'## {scenario} / {workload}', '',
                          '| Callback / metric | Median ms | p95 ms |', '|---|---:|---:|'])
            for callback, values in entry['callbacks'].items():
                metrics = values['metrics']
                for name in ('wallSeconds', 'managerPreparationSeconds', 'propagationWallSeconds',
                             'cachePreparationSeconds', 'eventSelectionSeconds',
                             *values['gpuStageRanking'], 'gradientWallSeconds',
                             'gradientModelPreparationSeconds', 'devicePropagationSeconds', 'gradientDeviceSeconds'):
                    if name in metrics:
                        value = metrics[name]
                        lines.append(f"| {callback} / {name} | {1000*value['median']:.4f} | {1000*value['p95']:.4f} |")
    lines.extend(['', 'GPU intervals overlap callback wall time; nested host timers also overlap.',
                  'Do not add all rows. Device intervals include stream gaps and are not hardware utilization.',
                  'Observables/search is fused; binning includes scans, sorting and allocation synchronization.',
                  'A gradient callback includes its own propagation. In valueThenGradient it usually reuses prediction.',
                  'Host remainder includes preparation, producer evaluation, penalties, launches and Python conversion.',
                  'See samples.jsonl for counts, transfer volumes and individual callback records.',
                  'See local-coordinate-costs.json for parameter-dependent migration and response costs.', ''])
    (output / 'report.md').write_text('\n'.join(lines))


def benchmark(args, report):
    import GUNDAM as G
    # Fail before the expensive dataset load if the installation is stale.
    required = PROPAGATION_FIELDS + COUNT_FIELDS + (GRADIENT_FIELDS if args.backend == 'CUDA' else ())
    missing = [name for name in required if not hasattr(G.BackendTimingSummary, name)]
    if missing:
        raise RuntimeError('Recompile and reinstall GUNDAM including PythonInterface; missing timers: ' + ', '.join(missing))
    report.update(module=G.__file__, nvidiaSmi=run_command(['nvidia-smi']))
    write_json(args.output / 'summary.json', report)
    G.setRuntimeWorkingDirectory(str(args.work_dir))
    G.setNumberOfThreads(args.threads)
    G.setLightOutputMode(True)
    G.FitterEngine.setRandomSeed(args.seed)
    builder = G.ConfigUtils.ConfigBuilder()
    builder.setConfig(args.config)
    for override in args.overrides:
        builder.override(override)
    builder.override(G.JsonType(json.dumps({'fitterEngineConfig': {
        'generateSamplePlots': False, 'generateOneSigmaPlots': False,
        'minimizerConfig': {'useAnalyticGradient': True, 'writeLlhHistory': False,
                            'enablePostFitErrorFit': False},
        'likelihoodInterfaceConfig': {'backendManagerConfig': {
            'isEnabled': True, 'useBackend': args.backend}}}})))
    resolved = json.loads(builder.toString())
    (args.output / 'resolved-config.json').write_text(builder.toString())
    normalized = resolved['fitterEngineConfig'].get('minimizerConfig', {}).get('useNormalizedFitSpace', True)
    reader = G.ConfigUtils.ConfigReader(builder.getConfig())
    reader.defineField(G.ConfigUtils.ConfigReader.FieldDefinition('fitterEngineConfig'))
    engine = G.FitterEngine()
    engine.configure(reader.fetchValueConfigReader('fitterEngineConfig'))
    likelihood = engine.getLikelihoodInterface()
    likelihood.setDataType(G.LikelihoodInterface.DataType.Asimov if args.asimov else G.LikelihoodInterface.DataType.RealData)
    likelihood.setForceAsimovData(args.asimov)
    minimizer = engine.getMinimizer()
    minimizer.setDisableCalcError(True)
    progress('Loading inputs ONCE and initializing; no minimization will run')
    start = time.perf_counter()
    engine.initialize()
    report['initializationSeconds'] = time.perf_counter() - start
    manager = engine.getBackendsManager()
    if not manager.hasBackend() or manager.getActiveBackendName() != args.backend:
        raise RuntimeError('Requested backend is not active: ' + args.backend)
    manager.setEnableAutoMaterialize(False)
    parameters = list(minimizer.getMinimizerFitParameterList())
    if not parameters:
        raise RuntimeError('No fit parameters')
    center, manifest, proposals = coordinate_manifest(parameters, normalized, args.amplitude)
    selected = [i for i, p in enumerate(manifest) if p['proposals'] and
                (not args.parameter_match or re.search(args.parameter_match, p['name']))]
    if not selected:
        raise RuntimeError('No variable coordinates match --parameter-match')
    write_json(args.output / 'parameters.json', dict(normalized=normalized, coordinates=manifest, selected=selected))
    report.update(events=likelihood.getModelPropagator().getSampleSet().getNbOfEvents(),
                  fitParameters=len(parameters), selectedParameters=len(selected), monitor=manager.getMonitorLine())
    if args.backend == 'CUDA' and hasattr(manager, 'getDiagnosticCuda'):
        report['cudaDiagnostics'] = str(manager.getDiagnosticCuda())
    write_json(args.output / 'summary.json', report)
    progress(f"Initialized in {report['initializationSeconds']:.1f} s; {report['events']} events, {len(parameters)} fit coordinates")

    def evaluate(callback, point):
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        result = minimizer.evalFit(point) if callback == 'value' else minimizer.evalFitGradientAt(point)
        wall, cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
        metrics = snapshot(manager, args.backend == 'CUDA' and callback == 'gradient')
        metrics.update(wallSeconds=wall, cpuSeconds=cpu)
        if callback == 'value':
            values = [float(result)]
        else:
            values = list(result)
            if len(values) != len(parameters):
                raise RuntimeError('Gradient dimension mismatch')
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError('Non-finite likelihood/gradient; reduce --amplitude')
        metrics['checksum'] = math.fsum(values)
        return metrics

    # Record first-use costs separately; gradient model construction can be substantial.
    report['firstValue'] = evaluate('value', center)
    report['firstGradient'] = evaluate('gradient', center)
    write_json(args.output / 'summary.json', report)
    rows = []
    with (args.output / 'samples.jsonl').open('w', buffering=1) as stream:
        for scenario in args.scenarios:
            count = (args.local_calls or len(selected)) if scenario == 'local1' else args.calls
            for workload in args.workloads:
                callbacks = ('value', 'gradient') if workload == 'valueThenGradient' else (workload,)
                minimizer.evalFit(center)
                for j in range(args.warmup):
                    point = list(center)
                    for i in selected:
                        point[i] = proposals[i][j % len(proposals[i])]
                    for callback in callbacks:
                        evaluate(callback, point)
                for trial in range(args.trials):
                    minimizer.evalFit(center)
                    for call in range(count):
                        point, index = list(center), None
                        if scenario == 'local1':
                            # Reset outside timing: each measured call changes one coordinate.
                            minimizer.evalFit(center)
                            index = selected[call % len(selected)]
                            point[index] = proposals[index][trial % len(proposals[index])]
                        elif scenario == 'global':
                            for slot, i in enumerate(selected):
                                point[i] = proposals[i][(slot + call + trial) % len(proposals[i])]
                        measured = {callback: evaluate(callback, point) for callback in callbacks}
                        row = dict(scenario=scenario, workload=workload, trial=trial, call=call,
                                   coordinate=index, callbacks=measured,
                                   wallSeconds=sum(value['wallSeconds'] for value in measured.values()),
                                   cpuSeconds=sum(value['cpuSeconds'] for value in measured.values()))
                        rows.append(row)
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                        if call % 50 == 0:
                            progress(f'{scenario}/{workload}: trial {trial+1}/{args.trials}, call {call+1}/{count}')
                    report['scenarios'] = summarize(rows, args.backend == 'CUDA')
                    write_json(args.output / 'summary.json', report)
                    write_report(args.output, report['scenarios'])
                    progress(f'{scenario}/{workload}: {1000*report["scenarios"][scenario][workload]["wallSeconds"]["median"]:.3f} ms')
    costs = []
    for workload in args.workloads:
        for i in selected:
            subset = [row for row in rows if row['coordinate'] == i and row['workload'] == workload]
            if subset:
                costs.append(dict(coordinate=i, name=manifest[i]['name'], workload=workload,
                                  profile=summarize(subset, args.backend == 'CUDA')['local1'][workload]))
    costs.sort(key=lambda entry: entry['profile']['wallSeconds']['median'], reverse=True)
    write_json(args.output / 'local-coordinate-costs.json', costs)
    report.update(complete=True, monitor=manager.getMonitorLine())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-of', '--override-files', dest='overrides', nargs='+', action='extend', default=[])
    parser.add_argument('-t', '--threads', type=int, default=4)
    parser.add_argument('-a', '--asimov', action='store_true')
    parser.add_argument('--work-dir', type=Path, default=Path.cwd())
    parser.add_argument('--backend', choices=['CUDA', 'CPU'], default='CUDA', help='CPU is for smoke testing; GPU timers require CUDA')
    parser.add_argument('--scenarios', nargs='+', choices=['constant', 'local1', 'global'], default=['constant', 'local1', 'global'])
    parser.add_argument('--workloads', nargs='+', choices=['value', 'gradient', 'valueThenGradient'], default=['value', 'gradient', 'valueThenGradient'])
    parser.add_argument('--calls', type=int, default=30, help='Calls per trial for constant/global')
    parser.add_argument('--local-calls', type=int, default=0, help='0 visits every selected fit coordinate per trial')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--amplitude', type=float, default=.01, help='Displacement in parameter standard deviations')
    parser.add_argument('--parameter-match', help='Regex selecting coordinates to perturb; full gradient still computed')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    if min(args.threads, args.calls, args.trials) < 1 or min(args.local_calls, args.warmup) < 0:
        parser.error('Counts must be positive; local-calls/warmup may be zero')
    if not math.isfinite(args.amplitude) or args.amplitude <= 0.:
        parser.error('--amplitude must be finite and positive')
    if args.parameter_match:
        try:
            re.compile(args.parameter_match)
        except re.error as error:
            parser.error(str(error))
    args.workloads, args.scenarios = list(dict.fromkeys(args.workloads)), list(dict.fromkeys(args.scenarios))
    args.work_dir, args.output = args.work_dir.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, settings={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  platform=platform.platform(), python=sys.version, pid=os.getpid(),
                  sourceRevision=run_command(['git', '-C', str(Path(__file__).resolve().parents[3]), 'describe', '--always', '--dirty']),
                  environment={key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_ID', 'OMP_NUM_THREADS')})
    write_json(args.output / 'summary.json', report)
    try:
        benchmark(args, report)
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        write_json(args.output / 'summary.json', report)
    progress(f'Complete: {args.output / "report.md"}')


if __name__ == '__main__':
    main()
