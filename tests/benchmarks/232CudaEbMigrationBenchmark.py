#!/usr/bin/env python3
"""Compare CPU/CUDA Eb composite/formula migration with hard and smoothed binning.

Only v13Prod, onlyEb and ebMigrationDial are applied by default. The benchmark
selects backends and configures hard/smooth binning itself; no backend or binResponse
override is needed. Workers load independent Asimov engines. No fit/Hesse is run.
"""
from array import array
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location(
    'cuda_profiling', Path(__file__).with_name('229CudaBackendProfiling.py'))
profiling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profiling)
write_json, progress = profiling.write_json, profiling.progress
DEFAULT_OVERRIDES = ['override/v13Prod.yaml', 'override/onlyEb.yaml', 'override/ebMigrationDial.yaml']
DEFAULT_CASES = ['CPU', 'CUDA-cachedResponses', 'CUDA-cachedResponses-interleaved',
                 'CUDA-recomputeResponses', 'CUDA-recomputeResponses-interleaved']


def prepare_config(config, variant, variable, width, sample_match):
    """Change explicit benchmark controls after resolving the three user overrides."""
    config = copy.deepcopy(config)
    samples = config['fitterEngineConfig']['likelihoodInterfaceConfig'][
        'propagatorConfig']['sampleSetConfig']['sampleList']
    if not isinstance(samples, list):
        raise RuntimeError('Expected an unfolded sampleList')
    selected = []
    for sample in samples:
        # Both variants start without pre-existing smoothing on other axes/samples.
        sample['binningResponse'] = {'isEnabled': False, 'variables': []}
        if not sample.get('isEnabled', True) or not re.search(sample_match, sample['name']):
            continue
        selected.append(sample['name'])
        # Let variable dials enable dynamic binning. Explicitly requesting it
        # marks samples as externally editable and forces host-variable polling,
        # defeating the dependency cache this benchmark is supposed to measure.
        sample['useDynamicBinning'] = False
        if variant == 'smooth':
            sample['binningResponse'] = {'isEnabled': True,
                'variables': [{'name': variable, 'width': width}]}
    if not selected:
        raise RuntimeError('No enabled sample matches --sample-match')
    counts = {'Composite': 0, 'Formula': 0}

    def visit(value):
        if isinstance(value, dict):
            if not value.get('isEnabled', True):
                return
            if value.get('dialType') in counts:
                counts[value['dialType']] += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(config['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig'])
    if not all(counts.values()):
        raise RuntimeError(f'The selected configuration needs Composite and Formula definitions: {counts}')
    return config, {'variant': variant, 'variable': variable, 'width': width if variant == 'smooth' else 0.,
                    'selectedSamples': selected, 'configuredDialDefinitions': counts,
                    'dynamicBinningSource': 'automatic variable-dial activation'}


def check_loaded_model(engine, coverage, case, probe_events):
    samples = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
    actual, fractional, probed = [], 0, 0
    for sample in samples:
        if not sample.isEnabled() or sample.getName() not in coverage['selectedSamples']:
            continue
        histogram, events = sample.getHistogram(), sample.getEventList()
        smooth = histogram.isBinningResponseEnabled()
        if smooth != (coverage['variant'] == 'smooth') or not sample.isUseDynamicBinning():
            raise RuntimeError(f'{sample.getName()}: requested binning mode was not activated')
        contexts = histogram.getBinContextList()
        if not contexts or not all(ctx.getBin().isVariableSet(coverage['variable']) for ctx in contexts):
            raise RuntimeError(f'{sample.getName()}: {coverage["variable"]} must be a binning axis; use --sample-match')
        actual.append({'name': sample.getName(), 'events': len(events), 'bins': len(contexts),
                       'dynamic': True, 'smoothed': smooth})
        if smooth:
            for index in range(min(len(events), probe_events)):
                event = events[index]
                response = histogram.evaluateEventBinningResponse(event)
                fractions = [part.fraction for part in response.binFractions] + [response.outsideFraction]
                if any(not math.isfinite(f) or f < -1e-12 or f > 1. + 1e-12 for f in fractions):
                    raise RuntimeError('Invalid reference binning fractions')
                if not math.isclose(sum(fractions), 1., rel_tol=0., abs_tol=1e-10):
                    raise RuntimeError('Reference binning fractions do not sum to one')
                fractional += any(1e-12 < f < 1. - 1e-12 for f in fractions)
                probed += 1
    if sum(s['events'] for s in actual) == 0:
        raise RuntimeError('No selected sample contains events')
    if coverage['variant'] == 'smooth' and fractional == 0:
        raise RuntimeError('No fractional bin response in the host probe; increase --probe-events or check --smoothing-width')
    coverage.update(samples=actual, hostFractionalProbe={'events': probed, 'fractionalEvents': fractional})
    if case != 'CPU':
        model = json.loads(engine.getBackendsManager().getDiagnosticCuda())
        if not model.get('available') or not model.get('hasEventState') or not model.get('observableResponses'):
            raise RuntimeError('CUDA event/observable propagation is not active')
        if not model.get('compositePrograms') or not model.get('formulaPrograms'):
            raise RuntimeError('CUDA Composite/Formula programs are absent; rebuild/select the current Python module')
        if model.get('hasSmoothedBinning') != (coverage['variant'] == 'smooth'):
            raise RuntimeError('CUDA packed binning does not match the requested variant')
        coverage['cudaModel'] = model


def read_snapshot(directory, entry):
    values = array('d')
    with (directory / entry['file']).open('rb') as stream:
        values.fromfile(stream, entry['values'])
        if stream.read(1):
            raise RuntimeError('Unexpected trailing snapshot data')
    return values


def migration_summary(directory, center, scenarios, trials):
    original = read_snapshot(directory, center)
    results = {}
    for scenario in scenarios:
        entries = []
        for trial in range(1, trials + 1):
            result = json.loads((directory / f'{scenario}-trial{trial}.json').read_text())
            snapshot = result['eventSnapshot']
            if any(snapshot[key] != center[key] for key in ('sampleEventCounts', 'stride', 'variables', 'values')):
                raise RuntimeError('Center/trial event layouts differ')
            values = read_snapshot(directory, snapshot)
            stride = center['stride']
            shifts = [abs(values[i] - original[i]) for i in range(2, len(values), stride)]
            entries.append({'trial': trial, 'eventsWithChangedObservable': sum(delta > 1e-12 for delta in shifts),
                'hardBinChanges': sum(values[i] != original[i] for i in range(1, len(values), stride)),
                'maxAbsoluteObservableShift': max(shifts, default=0.)})
        results[scenario] = entries
    if not any(e['eventsWithChangedObservable'] for scenario, entries in results.items()
               if scenario != 'constant' for e in entries):
        raise RuntimeError('No observable moved at the measured trial endpoints; check Eb coordinates/dials')
    return results


def worker(args):
    import GUNDAM
    started = time.monotonic()
    GUNDAM.setRuntimeWorkingDirectory(str(args.work_dir))
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfig(str(args.config))
    if args.override:
        builder.override(args.override)
    config, coverage = prepare_config(json.loads(builder.getConfig().toString()), args.worker_variant,
        args.variable, args.smoothing_width, args.sample_match)
    write_json(args.output / 'input-config.json', config)
    # Reuse the established timing/replay/comparison code, including resident
    # device buffers within this worker. Only the configuration is specialized.
    replay = argparse.Namespace(**vars(args))
    replay.config, replay.override = args.output / 'input-config.json', []
    replay.worker, replay.execution_mode = args.worker_case, 'isolated'
    replay.event_variables, replay.require_rebinning = [args.variable], True
    engine = profiling.load_engine(replay)
    sequence = profiling.parameter_sequence(replay, engine)
    center = profiling.snapshot_events(replay, engine, 'center', args.seed)
    check_loaded_model(engine, coverage, args.worker_case, args.probe_events)
    coverage.update(pythonModule=GUNDAM.__file__, parameters=sequence.manifest(), centerSnapshot=center)
    write_json(args.output / 'coverage.json', coverage)
    write_json(args.output / 'setup.json', {'seconds': time.monotonic() - started,
        'includesDatasetLoading': True, 'executionMode': 'isolated', 'pid': os.getpid()})
    profiling.measure(replay, engine, sequence)
    movement = migration_summary(args.output, center, args.scenarios, args.trials)
    write_json(args.output / 'migration.json', movement)
    progress(f'{args.worker_variant}/{args.worker_case}: coverage and migration verified')


def run_worker(args, variant, case):
    directory = args.output / variant / case
    directory.mkdir(parents=True)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:],
               '--worker-case', case, '--worker-variant', variant, '--output', str(directory)]
    with (directory / 'native.log').open('w') as log:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace')
        def forward():
            for line in child.stdout:
                log.write(line)
                log.flush()
                if '[benchmark]' in line:
                    print(f'[{variant}] {line.rstrip()}', flush=True)
        reader = threading.Thread(target=forward)
        reader.start()
        try:
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    progress(f'{variant}/{case}: worker running; {directory / "native.log"}')
        except BaseException:
            child.terminate()
            child.wait()
            raise
        finally:
            reader.join()
    if code:
        raise RuntimeError(f'{variant}/{case} failed ({code}); see {directory / "native.log"}')


def verify_variant_replay(args):
    """Check comparable inputs/parameter points without equating different models."""
    if len(args.variants) < 2:
        return
    reference = args.output / args.variants[0] / 'CPU'
    manifest = json.loads((reference / 'parameters.json').read_text())
    coverage = json.loads((reference / 'coverage.json').read_text())
    layout = [(s['name'], s['events'], s['bins']) for s in coverage['samples']]
    for variant in args.variants[1:]:
        directory = args.output / variant / 'CPU'
        if json.loads((directory / 'parameters.json').read_text()) != manifest:
            raise RuntimeError('Binning variants have different parameter manifests')
        other = json.loads((directory / 'coverage.json').read_text())
        if [(s['name'], s['events'], s['bins']) for s in other['samples']] != layout:
            raise RuntimeError('Binning variants loaded different selected samples/events')
        for scenario in args.scenarios:
            for trial in range(1, args.trials + 1):
                filename = f'{scenario}-trial{trial}.json'
                a = json.loads((reference / filename).read_text())['rows']
                b = json.loads((directory / filename).read_text())['rows']
                if [r['sequenceDigest'] for r in a] != [r['sequenceDigest'] for r in b]:
                    raise RuntimeError('Binning variants replayed different parameter points')


def write_report(args):
    summaries = {variant: {case: json.loads((args.output / variant / case / 'summary.json').read_text())
                           for case in args.cases} for variant in args.variants}
    lines = ['# Eb migration: CPU / CUDA', '',
             'Same seeded coordinates across cases and variants; independent Asimov engines.',
             'Times are medians; input loading, event readback, snapshots and numerical checks are excluded.', '',
             '| Binning | Case | Scenario | Eval ms | Responses ms | Variables/search ms | Binning/index ms | Histogram ms | GPU total ms | CPU speedup |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for variant, cases in summaries.items():
        for case, summary in cases.items():
            for scenario, entry in summary['scenarios'].items():
                t = entry['timings']
                fields = ['evaluationSeconds', 'cachedDialStageSeconds', 'variableStageSeconds',
                          'binningStageSeconds', 'histogramStageSeconds', 'devicePropagationSeconds']
                values = [t[f]['medianMs'] for f in fields]
                base = cases['CPU']['scenarios'][scenario]['timings']['evaluationSeconds']['medianMs']
                lines.append(f'| {variant} | {case} | {scenario} | ' +
                             ' | '.join(f'{v:.4f}' for v in values) + f' | {base / values[0]:.2f}x |')
    lines += ['', '## Migration coverage (CPU reference)', '',
              '| Binning | Scenario | Moved events (max endpoint) | Hard bin changes (max endpoint) | Max observable shift |',
              '|---|---|---:|---:|---:|']
    for variant in args.variants:
        movement = json.loads((args.output / variant / 'CPU' / 'migration.json').read_text())
        for scenario, entries in movement.items():
            lines.append(f'| {variant} | {scenario} | ' + ' | '.join(str(max(e[key] for e in entries)) for key in
                ('eventsWithChangedObservable', 'hardBinChanges', 'maxAbsoluteObservableShift')) + ' |')
    if 'hard' in summaries and 'smooth' in summaries:
        lines += ['', '## Smoothing overhead', '',
                  '| Case | Scenario | Smooth / hard evaluation | Delta evaluation ms | Delta binning/index ms | Delta histogram ms |',
                  '|---|---|---:|---:|---:|---:|']
        for case in args.cases:
            for scenario in args.scenarios:
                a = summaries['hard'][case]['scenarios'][scenario]['timings']
                b = summaries['smooth'][case]['scenarios'][scenario]['timings']
                ratio = b['evaluationSeconds']['medianMs'] / a['evaluationSeconds']['medianMs']
                deltas = [b[f]['medianMs'] - a[f]['medianMs'] for f in
                          ('evaluationSeconds', 'binningStageSeconds', 'histogramStageSeconds')]
                lines.append(f'| {case} | {scenario} | {ratio:.3f}x | ' +
                             ' | '.join(f'{d:+.4f}' for d in deltas) + ' |')
    lines += ['', 'Response time combines Composite traversal, Formula nodes and spline evaluation;',
              'the existing kernel timers do not separate those operations.',
              'CUDA variables/search includes the fused observable/hard-bin stage; binning/index',
              'includes fraction counting, scan and contribution sorting. CPU stage boundaries differ.',
              'Compare full evaluation first; GPU times overlap host evaluation and must not be added.',
              'rebinnedEventCount counts processed events, not hard bin changes. See migration.json',
              'for endpoint observable movement and hard bin changes; fractional probes are host checks.',
              'Each variant compares every measured likelihood and each trial endpoint histogram,',
              'outside bin, weight, observable and exact bin index against its own CPU baseline.',
              'Hard and smooth likelihoods are different models and are not compared numerically.',
              'Per-coordinate costs, cache candidates and GPU stage rankings are in each case directory.',
              'These synthetic workloads do not represent a complete fit or Hesse.', '']
    (args.output / 'report.md').write_text('\n'.join(lines))
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configOA2024.yaml'))
    parser.add_argument('--override', nargs='*', default=DEFAULT_OVERRIDES)
    parser.add_argument('--cases', nargs='+', choices=profiling.CASES, default=DEFAULT_CASES)
    parser.add_argument('--variants', nargs='+', choices=['hard', 'smooth'], default=['hard', 'smooth'])
    parser.add_argument('--variable', default='Pmu', help='Migrated binning variable, recorded in every event snapshot')
    parser.add_argument('--smoothing-width', type=float, default=1., help='Absolute width in variable units (Pmu: MeV)')
    parser.add_argument('--sample-match', default='.*', help='Regex for samples receiving dynamic/smoothed binning')
    parser.add_argument('--parameter-match', default='EB_', help='Regex selecting effective fit coordinates by full title')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--calls', type=int, default=100)
    parser.add_argument('--global-calls', type=int, default=40)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--probe-events', type=int, default=4096, help='Maximum events probed per sample for fractional responses')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--step-scale', type=float, default=.1)
    parser.add_argument('--scenarios', nargs='+', choices=['constant', 'local1', 'local2', 'global'],
                        default=['constant', 'local1', 'local2', 'global'])
    parser.add_argument('--relative-tolerance', type=float, default=1e-9)
    parser.add_argument('--absolute-tolerance', type=float, default=1e-7)
    parser.add_argument('--output', type=Path, default=Path(time.strftime('cuda-eb-migration-%Y%m%d-%H%M%S-') + str(os.getpid())))
    parser.add_argument('--worker-case', choices=profiling.CASES, help=argparse.SUPPRESS)
    parser.add_argument('--worker-variant', choices=['hard', 'smooth'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.threads, args.calls, args.global_calls, args.trials, args.probe_events) <= 0 or min(args.warmup, args.device_index) < 0:
        parser.error('Counts must be positive; warmup and device-index may be zero')
    for key in ('smoothing_width', 'step_scale', 'relative_tolerance', 'absolute_tolerance'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f'{key} must be finite and positive')
    for pattern in (args.parameter_match, args.sample_match):
        try:
            re.compile(pattern)
        except re.error as error:
            parser.error(f'Invalid regex: {error}')
    if 'CPU' not in args.cases:
        parser.error('--cases must include CPU for numerical validation')
    if not any(s != 'constant' for s in args.scenarios):
        parser.error('Select at least one changing scenario to exercise migration')
    if bool(args.worker_case) != bool(args.worker_variant):
        parser.error('Both internal worker arguments are required together')
    args.cases = ['CPU'] + list(dict.fromkeys(c for c in args.cases if c != 'CPU'))
    args.variants, args.scenarios = list(dict.fromkeys(args.variants)), list(dict.fromkeys(args.scenarios))
    args.work_dir = args.work_dir.resolve()
    args.config = (args.work_dir / args.config).resolve()
    args.output = args.output.resolve()
    if not args.config.is_file():
        parser.error(f'Configuration does not exist: {args.config}')
    return args


def main():
    args = parse_args()
    if args.worker_case:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(host=platform.node(), platform=platform.platform(), python=sys.executable,
                    environment={key: os.environ.get(key) for key in
                                 ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_ID', 'PYTHONPATH', 'LD_LIBRARY_PATH')},
                    scriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    profilingSha256=hashlib.sha256(Path(profiling.__file__).read_bytes()).hexdigest())
    settings['gitCommit'] = subprocess.run(['git', '-C', str(Path(__file__).resolve().parents[2]),
        'rev-parse', 'HEAD'], capture_output=True, text=True, check=False).stdout.strip()
    try:
        settings['nvidiaSmi'] = subprocess.run(['nvidia-smi'], capture_output=True, text=True,
                                              timeout=15, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        settings['nvidiaSmi'] = None
    write_json(args.output / 'settings.json', settings)
    write_json(args.output / 'summary.json', {'complete': False})
    for variant in args.variants:
        (args.output / variant).mkdir()
        write_json(args.output / variant / 'settings.json', {**settings, 'execution_mode': 'isolated', 'variant': variant})
        for case in args.cases:
            run_worker(args, variant, case)
        profiling.compare(args.output / variant, args.cases, args.scenarios, args.trials,
                          args.relative_tolerance, args.absolute_tolerance)
        profiling.write_report(args.output / variant, args.cases)
    verify_variant_replay(args)
    summaries = write_report(args)
    write_json(args.output / 'summary.json', {'complete': True, 'variants': summaries,
        'gpuComparisonPerformed': any(case != 'CPU' for case in args.cases)})
    progress(f'SUCCESS: {args.output / "report.md"}')


if __name__ == '__main__':
    main()
