#!/usr/bin/env python3
"""Profile resident production CUDA likelihood evaluations and compare response modes.

By default cases share one loaded Asimov model. No fit/Hesse is performed.
Backend activation also supports observable, dynamic and smoothed models.
Use --execution-mode isolated for an independent process and dataset load per case.
CUDA stages use the backend's existing stream events, without extra stage fences.
"""
from array import array
import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    'scheduling', Path(__file__).with_name('222CpuWeightSchedulingBenchmark.py'))
scheduling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduling)
progress, write_json = scheduling.progress, scheduling.write_json
CASES = ('CUDA-cachedResponses', 'CUDA-recomputeResponses', 'CPU',
         'CUDA-cachedResponses-interleaved', 'CUDA-recomputeResponses-interleaved',
         'CUDA-cachedResponses-interleaved-compact', 'CUDA-recomputeResponses-interleaved-compact')
SECONDS_FIELDS = ('parameterUploadSeconds', 'cachedDialStageSeconds',
                  'cachePreparationSeconds',
                  'eventWeightsStageSeconds', 'variableStageSeconds', 'binningStageSeconds',
                  'histogramStageSeconds', 'externalWeightUploadSeconds',
                  'devicePropagationSeconds', 'histogramReadbackSeconds',
                  'likelihoodHostSeconds', 'likelihoodDeviceSeconds', 'eventWeightReadbackSeconds')
COUNT_FIELDS = ('histogramReadbackBytes', 'eventWeightReadbackBytes', 'eventStateReadbackBytes',
                'cacheCandidateCount', 'rebinnedEventCount', 'binCandidateCount',
                'binContributionCount', 'contributionBufferBytes', 'externalWeightUploadBytes',
                'externalWeightUploadBlocks', 'likelihoodReadbackBytes')


def stats(values):
    ordered = sorted(values)
    return {'medianMs': statistics.median(values) * 1000,
            'meanMs': statistics.mean(values) * 1000,
            'p95Ms': ordered[int(.95 * (len(ordered) - 1))] * 1000}


def summarize(rows):
    return {name: stats([row['seconds'][name] for row in rows]) for name in rows[0]['seconds']}


def histogram(engine):
    engine.getBackendsManager().materialize('Histograms')
    return [(b.sumWeights, b.sqrtSumSqWeights)
            for sample in engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
            for b in sample.getHistogram().getBinContentList()]


def outside_histogram(engine):
    return [(sample.getHistogram().getOutsideBinContent().sumWeights,
             sample.getHistogram().getOutsideBinContent().sqrtSumSqWeights)
            for sample in engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
            if sample.getHistogram().isBinningResponseEnabled()]


def parameter_sequence(args, engine):
    sequence = scheduling.ParameterSequence(engine, args.step_scale)
    if args.parameter_match:
        pattern = re.compile(args.parameter_match)
        sequence.entries = [entry for entry in sequence.entries if pattern.search(entry[0].getFullTitle())]
        if not sequence.entries:
            raise RuntimeError(f'No eligible effective fit coordinate matches {args.parameter_match!r}')
        sequence.current = [center for _, center, _, _ in sequence.entries]
    if 'local2' in args.scenarios and len(sequence.entries) < 2:
        raise RuntimeError('local2 needs two coordinates; use --scenarios constant local1 for a single parameter')
    progress(f'{args.worker}: {len(sequence.entries)} selected fit coordinates')
    if args.parameter_match:
        for entry in sequence.manifest():
            progress(f"  {entry['name']}: center {entry['center']}, proposal sigma {entry['proposalStdDev']}")
    return sequence


def count_summary(rows):
    return {name: {'median': statistics.median(row['counts'][name] for row in rows),
                   'max': max(row['counts'][name] for row in rows)} for name in COUNT_FIELDS}


def backend_config(args):
    backend_type = 'CPU' if args.worker == 'CPU' else 'CUDA'
    options = {} if backend_type == 'CPU' else {
        'deviceIndex': args.device_index, 'propagationMode': args.worker.split('-')[1],
        'weightLayout': 'warpInterleaved' if '-interleaved' in args.worker else 'eventMajor',
        'likelihoodBlockSize': getattr(args, 'likelihood_block_size', 0),
        'reusePrediction': not getattr(args, 'disable_prediction_reuse', False)}
    if args.worker.endswith('-compact'):
        options['compactDials'] = True
    return {'isEnabled': True, 'useBackend': backend_type,
        'options': [{'name': backend_type, **options}]}


def load_engine(args, *, preserve_configuration=False):
    import GUNDAM
    progress(f'{args.worker}: Python module {GUNDAM.__file__}')
    missing = [name for name in (*SECONDS_FIELDS, *COUNT_FIELDS)
               if not hasattr(GUNDAM.BackendTimingSummary, name)]
    if missing:
        raise RuntimeError(f'Rebuild/select the matching Python interface: missing {missing}')
    GUNDAM.setRuntimeWorkingDirectory(str(args.work_dir))
    GUNDAM.setNumberOfThreads(args.threads)
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setIsDebug(False)
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfig(str(args.config))
    if args.override:
        builder.override(args.override)
    config = json.loads(builder.getConfig().toString())['fitterEngineConfig']
    if not preserve_configuration:
        config.update(enablePca=False, scaleParStepWithChi2Response=False,
                      generateSamplePlots=False, generateOneSigmaPlots=False)
    probability = getattr(args, 'likelihood', 'configured')
    if probability != 'configured':
        llh = config.setdefault('likelihoodInterfaceConfig', {})
        legacy = config.get('propagatorConfig', {})
        samples = legacy.get('sampleSetConfig', legacy.get('fitSampleSetConfig', {}))
        original = llh.get('jointProbabilityConfig', legacy.get('llhConfig', samples.get('llhConfig', {})))
        barlow_options = {'ignoreBinsWithZeroPredictionAtPrior', 'muteZeroPredBinWarn',
            'allowZeroMcWhenZeroData', 'usePoissonLikelihood', 'BBNoUpdateWeights',
            'fractionalErrorLimit', 'expectedValueMinimum', 'verboseLevel', 'isVerbose', 'throwIfInfLlh'}
        llh['jointProbabilityConfig'] = {k: v for k,v in original.items() if k in barlow_options} if probability != 'PoissonLLH' else {}
        llh['jointProbabilityConfig']['type'] = probability
        if probability == 'PoissonLLH':
            llh['jointProbabilityConfig']['ignoreBinsWithZeroPredictionAtPrior'] = True
    # Replace backend-specific overrides so CPU never inherits CUDA-only keys.
    backend = backend_config(args)
    backend_type = backend['useBackend']
    config.setdefault('likelihoodInterfaceConfig', {})['backendManagerConfig'] = backend
    write_json(args.output / 'resolved-fitter-config.json', config)
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    progress('Loading dataset once; input selection/limits are preserved')
    start = time.monotonic()
    engine.initialize()
    manager = engine.getBackendsManager()
    manager.setEnableAutoMaterialize(False)
    if not manager.hasBackend() or manager.getType() != backend_type:
        raise RuntimeError(f'Expected an active {backend_type} backend')
    engine.evaluateLikelihood()
    progress(f'{args.worker}: initialized in {time.monotonic() - start:.1f} s')
    return engine


def snapshot_events(args, engine, scenario, seed):
    """Materialize only after evaluation timers; retain values for tolerant comparisons."""
    manager = engine.getBackendsManager()
    # A final untimed evaluation uses normal materialization, which also handles
    # fixed-bin models that do not publish an EventState output.
    manager.setEnableAutoMaterialize(True)
    try:
        engine.evaluateLikelihood()
    finally:
        manager.setEnableAutoMaterialize(False)
    values, counts = array('d'), []
    samples = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
    for sample in samples:
        events = sample.getEventList()
        counts.append(len(events))
        for event in events:
            values.extend((event.getEventWeight(), float(event.getIndices().bin)))
            for name in args.event_variables:
                values.append(event.getVariables().fetchVariable(name).getVarAsDouble())
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError('Nonfinite materialized event state')
    filename = f'{scenario}-events-{seed}.bin'
    with (args.output / filename).open('wb') as stream:
        values.tofile(stream)
    return {'file': filename, 'sampleEventCounts': counts, 'variables': args.event_variables,
            'stride': 2 + len(args.event_variables), 'values': len(values), 'format': 'native-double'}


def compare_events(left_dir, right_dir, left, right, relative, absolute):
    if left is None and right is None:
        return 0., 0
    if left is None or right is None or any(left[key] != right[key] for key in
            ('sampleEventCounts', 'variables', 'stride', 'values', 'format')):
        raise RuntimeError('Event snapshot layouts differ')
    snapshots = []
    for directory, entry in ((left_dir, left), (right_dir, right)):
        values = array('d')
        with (directory / entry['file']).open('rb') as stream:
            values.fromfile(stream, entry['values'])
            if stream.read(1):
                raise RuntimeError('Unexpected trailing event snapshot data')
        snapshots.append(values)
    maximum, mismatches = 0., 0
    for index, (a, b) in enumerate(zip(*snapshots)):
        if index % left['stride'] == 1:
            mismatches += a != b  # Bin membership must agree exactly.
        else:
            maximum = max(maximum, abs(a - b))
            if not math.isclose(a, b, rel_tol=relative, abs_tol=absolute):
                raise RuntimeError(f'Event state differs at value {index}: {a} versus {b}')
    return maximum, mismatches


class CudaProfiler:
    """Explicit capture gates; imported only for a requested hardware profile."""
    def __init__(self, library, device):
        import ctypes
        import ctypes.util
        self.runtime = ctypes.CDLL(library or ctypes.util.find_library('cudart') or 'libcudart.so')
        for name in ('cudaProfilerStart', 'cudaProfilerStop'):
            getattr(self.runtime, name).restype = ctypes.c_int
            getattr(self.runtime, name).argtypes = []
        self.runtime.cudaSetDevice.argtypes = [ctypes.c_int]
        self.check(self.runtime.cudaSetDevice(device), 'cudaSetDevice')

    @staticmethod
    def check(code, action):
        if code:
            raise RuntimeError(f'{action} failed with CUDA error {code}')

    def start(self):
        self.check(self.runtime.cudaProfilerStart(), 'cudaProfilerStart')

    def stop(self):
        self.check(self.runtime.cudaProfilerStop(), 'cudaProfilerStop')


def evaluate_pass(args, engine, sequence, scenario, calls, seed, *, capture=False):
    sequence.reset()
    engine.evaluateLikelihood()  # Identical center and primed cache in every trial.
    rng, digest, rows = random.Random(seed), hashlib.sha256(), []
    manager = engine.getBackendsManager()
    last_progress = time.monotonic()
    for call in range(calls):
        indices = list(sequence.coordinates(scenario, call))
        start = time.perf_counter()
        changed = sequence.apply(scenario, rng, call, digest)
        update_seconds = time.perf_counter() - start
        selected = capture and call == args.profile_call
        if selected:
            progress(f'NCU capture: {args.worker}/{scenario}, call {call}, coordinates {indices}')
            args.profiler.start()
        start = time.perf_counter()
        try:
            engine.evaluateLikelihood()
        finally:
            if selected:
                args.profiler.stop()
        elapsed = time.perf_counter() - start
        value = engine.getLikelihoodInterface().getLastLikelihood()
        timing = manager.getLastTimingSummary()
        seconds = {name: getattr(timing, name) for name in SECONDS_FIELDS}
        seconds.update(evaluationSeconds=elapsed, parameterUpdateSeconds=update_seconds,
                       hostPreparationSeconds=manager.getLastPreparationSeconds())
        if not math.isfinite(value) or any(not math.isfinite(v) or v < 0 for v in seconds.values()):
            raise RuntimeError(f'{scenario}: nonfinite result or invalid timing at call {call}')
        counts = {name: getattr(timing, name) for name in COUNT_FIELDS}
        if args.worker != 'CPU' and getattr(args, 'require_device_likelihood', False):
            if counts['histogramReadbackBytes'] or counts['likelihoodReadbackBytes'] != 8:
                raise RuntimeError('Expected a resident statistical likelihood with exactly 8 bytes of LLH readback')
        if counts['eventWeightReadbackBytes'] or counts['eventStateReadbackBytes']:
            raise RuntimeError('Unexpected event readback during resident evaluation')
        rows.append({'seconds': seconds, 'counts': counts, 'likelihood': value,
                     'statLikelihood': engine.getLikelihoodInterface().getLastStatLikelihood(),
                     'changedFitCoordinates': changed, 'coordinateIndices': indices,
                     'sequenceDigest': digest.hexdigest(), 'hardwareProfiled': selected})
        if time.monotonic() - last_progress >= 5:
            progress(f'{args.worker}/{scenario}: {call + 1}/{calls} evaluations')
            last_progress = time.monotonic()
    bins = histogram(engine)  # Explicit materialization is outside all evaluation timers.
    outside = outside_histogram(engine)
    if any(not math.isfinite(v) for pair in bins + outside for v in pair):
        raise RuntimeError('Nonfinite materialized histogram')
    result = {'rows': rows, 'timings': summarize(rows), 'histogram': bins, 'outsideHistogram': outside}
    if args.event_variables is not None:
        result['eventSnapshot'] = snapshot_events(args, engine, scenario, seed)
    return result


def measure(args, engine=None, sequence=None):
    import GUNDAM
    if engine is None:
        start = time.monotonic()
        engine = load_engine(args)
        write_json(args.output / 'setup.json', {
            'seconds': time.monotonic() - start, 'includesDatasetLoading': True,
            'executionMode': args.execution_mode, 'pid': os.getpid()})
    if sequence is None:
        sequence = parameter_sequence(args, engine)
    manifest = sequence.manifest()
    write_json(args.output / 'parameters.json', manifest)
    result = {'complete': False, 'case': args.worker, 'pythonModule': GUNDAM.__file__,
              'pythonModuleSha256': hashlib.sha256(Path(GUNDAM.__file__).read_bytes()).hexdigest(),
              'parameters': manifest, 'executionMode': args.execution_mode, 'pid': os.getpid(),
              'scenarios': {}, 'hardwareProfile': getattr(args, 'profile_capture', False)}
    if args.worker != 'CPU':
        result['model'] = json.loads(engine.getBackendsManager().getDiagnosticCuda())
        if args.worker.endswith('-compact') and not result['model'].get('compactDials', False):
            raise RuntimeError('Compact CUDA dials were requested but are inactive: ' + result['model'].get('compactDialFallback', 'rebuild the module'))
        if getattr(args, 'require_weight_inventory', False) and 'weightInventory' not in result['model']:
            raise RuntimeError('Rebuild/reinstall the CUDA Python module: weightInventory is missing')
        if getattr(args, 'require_device_likelihood', False) and result['model'].get('likelihoodExecution') != 'device':
            raise RuntimeError('This benchmark requires PoissonLLH or Barlow-Beeston 2022 resident on CUDA')
        if getattr(args, 'require_v100', False) and 'V100' not in result['model'].get('device', ''):
            raise RuntimeError('Selected CUDA device is not a V100: ' + result['model'].get('device', 'unknown'))
        write_json(args.output / 'model.json', result['model'])
        progress(f"{args.worker}: weight layout {result['model']['weightLayout']}, "
                 f"{result['model']['packedEventDialIndexBytes']} index bytes")
    for scenario in args.scenarios:
        calls = scheduling.scenario_calls(args, scenario, len(sequence.entries))
        if args.warmup:
            evaluate_pass(args, engine, sequence, scenario, args.warmup, args.seed + 1000000)
        trials = []
        for trial in range(args.trials):
            progress(f'{args.worker}/{scenario}: trial {trial + 1}/{args.trials}, {calls} calls')
            capture = getattr(args, 'profile_capture', False) and trial == 0 and args.worker != 'CPU'
            if capture and args.profile_call >= calls:
                raise RuntimeError(f'--profile-call {args.profile_call} exceeds {calls} calls for {scenario}')
            measurement = evaluate_pass(args, engine, sequence, scenario, calls, args.seed + trial, capture=capture)
            write_json(args.output / f'{scenario}-trial{trial + 1}.json', measurement)
            trials.append(measurement)
        rows = [row for trial in trials for row in trial['rows']]
        timings = summarize(rows)
        # CUDA event durations overlap the synchronous host evaluation. Rank stages
        # within each clock domain; never add GPU time to host wait/evaluation.
        cuda_stages = ('parameterUploadSeconds', 'cachedDialStageSeconds',
                       'eventWeightsStageSeconds', 'variableStageSeconds',
                       'binningStageSeconds', 'histogramStageSeconds', 'likelihoodDeviceSeconds')
        gpu_ranking = sorted(cuda_stages, key=lambda key: timings[key]['medianMs'], reverse=True)
        entry = {'callsPerTrial': calls, 'trials': args.trials, 'timings': timings,
                 'medianResponseCandidates': statistics.median(row['counts']['cacheCandidateCount'] for row in rows),
                 'gpuStageRanking': gpu_ranking if args.worker != 'CPU' else [],
                 'counts': count_summary(rows),
                 'coordinateCostsFile': None}
        if scenario == 'local1':
            grouped = [[] for _ in sequence.entries]
            for row in rows:
                grouped[row['coordinateIndices'][0]].append(row)
            costs = [{**manifest[index], 'samples': len(group),
                      'medianResponseCandidates': statistics.median(row['counts']['cacheCandidateCount'] for row in group),
                      'timings': summarize(group), 'counts': count_summary(group)} for index, group in enumerate(grouped)]
            costs.sort(key=lambda row: row['timings']['evaluationSeconds']['medianMs'], reverse=True)
            entry['coordinateCostsFile'] = 'local1-coordinate-costs.json'
            write_json(args.output / entry['coordinateCostsFile'], costs)
        if args.require_rebinning and scenario != 'constant' and not any(
                row['counts']['rebinnedEventCount'] > 0 for row in rows):
            raise RuntimeError(f'{args.worker}/{scenario}: no event was rebinned; check parameter selection and observable dials')
        result['scenarios'][scenario] = entry
        write_json(args.output / 'summary.json', result)
        progress(f"{args.worker}/{scenario}: evaluation {timings['evaluationSeconds']['medianMs']:.3f} ms; "
                 f"preparation {timings['hostPreparationSeconds']['medianMs']:.3f} ms; "
                 f"GPU {timings['devicePropagationSeconds']['medianMs']:.3f} ms; "
                 f"response candidates {entry['medianResponseCandidates']}; "
                 f"rebinned {entry['counts']['rebinnedEventCount']['median']}; "
                 f"contributions {entry['counts']['binContributionCount']['median']}")
    result['complete'] = True
    write_json(args.output / 'summary.json', result)


def measure_resident(args, state=None):
    import GUNDAM
    if not hasattr(GUNDAM.BackendManager, 'activate'):
        raise RuntimeError('Rebuild/reinstall the Python interface: activate is unavailable')
    if getattr(args, 'profile_capture', False):
        args.profiler = CudaProfiler(args.cuda_runtime, args.device_index)
    root = args.output
    engine, sequence, config = state if state is not None else (None, None, None)
    for case in args.cases:
        args.worker = case
        args.output = root / case
        args.output.mkdir()
        start = time.monotonic()
        loads_dataset = engine is None
        if loads_dataset:
            engine = load_engine(args)
            sequence = parameter_sequence(args, engine)
            config = json.loads((args.output / 'resolved-fitter-config.json').read_text())
        else:
            sequence.reset()
            backend = backend_config(args)
            progress(f'{case}: replacing backend; keeping loaded events, dials and Asimov data')
            engine.getBackendsManager().activate(
                backend['useBackend'], GUNDAM.JsonType(json.dumps(
                    {key: value for key, value in backend['options'][0].items() if key != 'name'})))
            config.setdefault('likelihoodInterfaceConfig', {})['backendManagerConfig'] = backend
            write_json(args.output / 'resolved-fitter-config.json', config)
            engine.evaluateLikelihood()  # Prime new buffers outside measurements.
        write_json(args.output / 'setup.json', {
            'seconds': time.monotonic() - start, 'includesDatasetLoading': loads_dataset,
            'executionMode': 'resident', 'pid': os.getpid()})
        measure(args, engine, sequence)
    args.output = root
    return engine, sequence, config


def compare(output, cases, scenarios, trials, relative, absolute, *, reference_directory=None,
            comparison_name='comparison.json'):
    reference = 'CPU' if 'CPU' in cases else cases[0]
    reference_dir = reference_directory if reference_directory is not None else output / reference
    manifest = json.loads((reference_dir / 'parameters.json').read_text())
    comparison = {'reference': reference, 'relativeTolerance': relative,
                  'absoluteTolerance': absolute, 'passed': True, 'cases': {}}
    if reference_directory is not None:
        comparison['referenceDirectory'] = str(reference_dir)
    for case in (case for case in cases if case != reference or reference_directory is not None):
        if manifest != json.loads((output / case / 'parameters.json').read_text()):
            raise RuntimeError(f'{case}: parameter manifests differ')
        entries = {}
        for scenario in scenarios:
            max_llh, max_bin, max_outside, passed = 0., 0., 0., True
            max_stat = 0.
            max_event, bin_mismatches = 0., 0
            for trial in range(1, trials + 1):
                name = f'{scenario}-trial{trial}.json'
                left = json.loads((reference_dir / name).read_text())
                right = json.loads((output / case / name).read_text())
                event_delta, mismatches = compare_events(reference_dir, output / case,
                    left.get('eventSnapshot'), right.get('eventSnapshot'), relative, absolute)
                max_event = max(max_event, event_delta)
                bin_mismatches += mismatches
                passed &= mismatches == 0
                if len(left['rows']) != len(right['rows']):
                    raise RuntimeError('Evaluation counts differ')
                for a, b in zip(left['rows'], right['rows']):
                    if a['sequenceDigest'] != b['sequenceDigest']:
                        raise RuntimeError('Parameter sequences differ')
                    if 'statLikelihood' in a and 'statLikelihood' in b:
                        max_stat = max(max_stat, abs(a['statLikelihood'] - b['statLikelihood']))
                        passed &= math.isclose(a['statLikelihood'], b['statLikelihood'], rel_tol=relative, abs_tol=absolute)
                    max_llh = max(max_llh, abs(a['likelihood'] - b['likelihood']))
                    passed &= math.isclose(a['likelihood'], b['likelihood'], rel_tol=relative, abs_tol=absolute)
                if len(left['histogram']) != len(right['histogram']):
                    raise RuntimeError('Histogram sizes differ')
                for a, b in zip(left['histogram'], right['histogram']):
                    for x, y in zip(a, b):
                        max_bin = max(max_bin, abs(x - y))
                        passed &= math.isclose(x, y, rel_tol=relative, abs_tol=absolute)
                left_outside, right_outside = left.get('outsideHistogram', []), right.get('outsideHistogram', [])
                if len(left_outside) != len(right_outside):
                    raise RuntimeError('Outside histogram sizes differ')
                for a, b in zip(left_outside, right_outside):
                    for x, y in zip(a, b):
                        max_outside = max(max_outside, abs(x - y))
                        passed &= math.isclose(x, y, rel_tol=relative, abs_tol=absolute)
            def median(case_name):
                summary = json.loads((output / case_name / 'summary.json').read_text())
                return summary['scenarios'][scenario]['timings']['evaluationSeconds']['medianMs']
            entries[scenario] = {'passed': passed, 'maxAbsoluteEventDifference': max_event,
                                 'eventBinMismatches': bin_mismatches, 'maxAbsoluteLikelihoodDifference': max_llh,
                                 'maxAbsoluteStatLikelihoodDifference': max_stat,
                                 'maxAbsoluteHistogramOrErrorDifference': max_bin,
                                 'maxAbsoluteOutsideOrErrorDifference': max_outside,
                                 'speedupOverReference': json.loads((reference_dir / 'summary.json').read_text())[
                                     'scenarios'][scenario]['timings']['evaluationSeconds']['medianMs'] / median(case)}
            comparison['passed'] &= passed
        comparison['cases'][case] = entries
    write_json(output / comparison_name, comparison)
    if not comparison['passed']:
        raise RuntimeError(f'Numerical comparison failed; see {comparison_name}')


def write_report(output, cases):
    lines = ['# Likelihood profiling', '',
             'Execution mode: ' + json.loads((output / 'settings.json').read_text())['execution_mode'] + '.', '',
             'All durations below are medians in milliseconds. GPU event intervals overlap',
             'host evaluation time. Do not add these columns together.', '',
             '| Case | Scenario | Evaluation | Host preparation | GPU total | Dial responses | Event weights | Observables + search | Bin index | Histograms | Histogram readback | Host likelihood | Device likelihood |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fields = ('evaluationSeconds', 'hostPreparationSeconds', 'devicePropagationSeconds',
              'cachedDialStageSeconds', 'eventWeightsStageSeconds', 'variableStageSeconds',
              'binningStageSeconds', 'histogramStageSeconds',
              'histogramReadbackSeconds', 'likelihoodHostSeconds', 'likelihoodDeviceSeconds')
    for case in cases:
        summary = json.loads((output / case / 'summary.json').read_text())
        for scenario, entry in summary['scenarios'].items():
            times = entry['timings']
            cells = [f'{times[key]["medianMs"]:.4f}' for key in fields]
            lines.append('| ' + ' | '.join([case, scenario, *cells]) + ' |')
    lines.extend(['', '| Case | Scenario | Rebin events (median/max) | Contributions (median) | Contribution capacity MiB (max) |',
                  '|---|---|---:|---:|---:|'])
    for case in cases:
        summary = json.loads((output / case / 'summary.json').read_text())
        for scenario, entry in summary['scenarios'].items():
            counts = entry['counts']
            rebin = counts['rebinnedEventCount']
            lines.append(f"| {case} | {scenario} | {rebin['median']}/{rebin['max']} | "
                         f"{counts['binContributionCount']['median']} | "
                         f"{counts['contributionBufferBytes']['max'] / 2**20:.2f} |")
    lines.extend(['', 'CUDA observables/search is the fused variable/hard-bin stage; bin index includes',
                  'fraction counting, prefix scan, allocation synchronization and contribution sorting.',
                  'CPU variable/binning stages have different boundaries. Compare total evaluations first.',
                  'Rebin events count a binning pass, not necessarily a change of hard bin.', '',
                  'CPU stage durations in the responses/weights/histograms columns are host durations.',
                  'GPU total is zero for CPU. CUDA upload/reset duration is in the JSON reports.', '',
                  'Case summaries and coordinate costs include `medianResponseCandidates`.',
                  'For CUDA this counts dials visited by the response kernels, before transformed-input reuse.',
                  'Primed constant cached evaluations should have zero candidates.',
                  'Host sparse-selection time is `cachePreparationSeconds`, inside full evaluation time.', '',
                  'Inspect `local1-coordinate-costs.json` for the most expensive parameter changes.',
                  'Each case summary ranks CUDA stages in `gpuStageRanking`.',
                  '`comparison.json` checks every measured likelihood and each trial\'s final histogram',
                  '(contents, errors and smoothed outside bins), and reports speedups against CPU when selected.',
                  '`--event-variables Pmu` also compares all event weights, Pmu and exact bin membership',
                  'after each trial. Event materialization and comparison are outside evaluation timers.', '',
                  'These are seeded propagation workloads, not the distribution of calls in a real fit.',
                  'The gap between evaluation and GPU time also contains host preparation, transfers,',
                  'likelihood/penalty work, launch overhead and synchronization. It is not all GPU wait.',
                  'No bandwidth, occupancy or hardware-counter claims are inferred from these timers.', ''])
    lines.extend(['| CUDA case | Requested layout | Actual layout | Index bytes |',
                  '|---|---|---|---:|'])
    for case in cases:
        model = json.loads((output / case / 'summary.json').read_text()).get('model')
        if model:
            lines.append(f"| {case} | {model['requestedWeightLayout']} | {model['weightLayout']} | "
                         f"{model['packedEventDialIndexBytes']} |")
    lines.extend(['', '| CUDA case | Compact requested | Compact active | Device descriptors bytes | Shared policies | Fallback |',
                  '|---|---|---|---:|---:|---|'])
    for case in cases:
        model = json.loads((output / case / 'summary.json').read_text()).get('model')
        if model:
            lines.append(f"| {case} | {model.get('requestedCompactDials', False)} | {model.get('compactDials', False)} | "
                         f"{model.get('deviceDialDescriptorBytes', 'unknown')} | {model.get('compactDialPolicies', 0)} | "
                         f"{model.get('compactDialFallback', '')} |")
    (output / 'report.md').write_text('\n'.join(lines))


def run_worker(args, case):
    directory = args.output if case == 'resident' else args.output / case
    if case != 'resident':
        directory.mkdir()
    log_path = directory / 'native.log'
    command = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:],
               '--worker', case, '--output', str(directory)]
    with log_path.open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace')
        def forward():
            for line in child.stdout:
                log.write(line)
                log.flush()
                if '[benchmark]' in line:
                    print(line.rstrip(), flush=True)
        reader = threading.Thread(target=forward)
        reader.start()
        try:
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    progress(f'{case}: worker PID {child.pid} running; log {log_path}')
        except KeyboardInterrupt:
            child.terminate()
            child.wait()
            raise
        finally:
            reader.join()
    if code:
        raise RuntimeError(f'{case} exited with code {code}; see {log_path}')


def initialize_output(args):
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(host=platform.node(), platform=platform.platform(), python=sys.executable,
                    allowedCpus=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                    environment={key: os.environ.get(key) for key in
                                 ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_ID', 'PYTHONPATH')})
    settings['gitCommit'] = subprocess.run(
        ['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'],
        capture_output=True, text=True, check=False).stdout.strip()
    try:
        settings['nvidiaSmi'] = subprocess.run(['nvidia-smi'], capture_output=True, text=True,
                                              timeout=15, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        settings['nvidiaSmi'] = None
    write_json(args.output / 'settings.json', settings)
    write_json(args.output / 'summary.json', {'complete': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--override', nargs='*', default=[])
    parser.add_argument('--work-dir', type=Path, default=Path.cwd())
    parser.add_argument('--cases', nargs='+', choices=CASES,
                        default=['CUDA-cachedResponses-interleaved', 'CUDA-recomputeResponses-interleaved'])
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--likelihood-block-size', type=int, choices=[0, 32, 64, 128, 256], default=0,
                        help='CUDA likelihood block size; zero selects the automatic policy')
    parser.add_argument('--disable-prediction-reuse', action='store_true',
                        help='Recompute weights/histograms/LLH even when responses are cached and unchanged')
    parser.add_argument('--likelihood', choices=['configured', 'PoissonLLH', 'BarlowLLH_BANFF_OA2021'], default='configured')
    parser.add_argument('--require-device-likelihood', action='store_true')
    parser.add_argument('--require-weight-inventory', action='store_true')
    parser.add_argument('--require-v100', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--calls', type=int, default=30)
    parser.add_argument('--global-calls', type=int, default=10)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--step-scale', type=float, default=.1)
    parser.add_argument('--parameter-match', help='Regex selecting effective fit coordinates by full title')
    parser.add_argument('--require-rebinning', action='store_true',
                        help='Require at least one measured rebin in every nonconstant scenario/case')
    parser.add_argument('--scenarios', nargs='+', choices=['constant', 'local1', 'local2', 'global'],
                        default=['constant', 'local1', 'local2', 'global'])
    parser.add_argument('--event-variables', nargs='*', default=None,
                        help='Compare all event weights, exact bin indices and these variables outside timers')
    parser.add_argument('--relative-tolerance', type=float, default=1e-9)
    parser.add_argument('--absolute-tolerance', type=float, default=1e-7)
    parser.add_argument('--output', type=Path,
                        default=Path(f'benchmark-{Path(__file__).stem}-{time.strftime("%Y%m%d%H%M%S")}'),
                        help='Output directory (default: ./benchmark-<script-name>-YYYYMMDDHHmmss)')
    parser.add_argument('--profile-capture', action='store_true',
                        help='Gate Nsight Compute around one call per CUDA case/scenario, after warmup; timings are perturbed')
    parser.add_argument('--profile-call', type=int, default=0, help='Zero-based measured call to capture in first trial')
    parser.add_argument('--cuda-runtime', help='Path to the matching libcudart.so for profiler gates')
    parser.add_argument('--execution-mode', choices=['resident', 'isolated'], default='resident')
    parser.add_argument('--worker', choices=(*CASES, 'resident'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.profile_call < 0:
        parser.error('--profile-call must be nonnegative')
    if args.profile_capture and (args.execution_mode != 'resident' or args.worker not in (None, 'resident')):
        parser.error('--profile-capture requires resident execution')
    if args.profile_capture and not args.warmup:
        parser.error('--profile-capture requires at least one warmup call')
    if min(args.threads, args.calls, args.global_calls, args.trials) <= 0 or min(args.warmup, args.device_index) < 0:
        parser.error('Counts must be positive; warmup and device-index may be zero')
    for name in ('step_scale', 'relative_tolerance', 'absolute_tolerance'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} must be finite and positive')
    if args.parameter_match:
        try:
            re.compile(args.parameter_match)
        except re.error as error:
            parser.error(f'Invalid --parameter-match: {error}')
    args.cases = list(dict.fromkeys(args.cases))
    args.scenarios = list(dict.fromkeys(args.scenarios))
    args.work_dir = args.work_dir.resolve()
    args.config = (args.work_dir / args.config).resolve()
    args.output = args.output.resolve()
    if args.worker:
        if args.worker == 'resident':
            measure_resident(args)
        else:
            measure(args)
        return
    if not args.config.is_file():
        parser.error(f'Configuration does not exist: {args.config}')
    initialize_output(args)
    if args.execution_mode == 'resident':
        run_worker(args, 'resident')
    else:
        for case in args.cases:
            run_worker(args, case)
    compare(args.output, args.cases, args.scenarios, args.trials,
            args.relative_tolerance, args.absolute_tolerance)
    write_report(args.output, args.cases)
    write_json(args.output / 'summary.json', {
        'complete': True, 'cases': {case: json.loads((args.output / case / 'summary.json').read_text())
                                  for case in args.cases}})
    progress(f'SUCCESS: {args.output}')


if __name__ == '__main__':
    main()
