#!/usr/bin/env python3
"""OA2024 graph search A/B: run once per build, then compare to --reference.

Requires a GPU allocation and a CUDA/Python-enabled Release build. Extra options
are forwarded to 229CudaBackendProfiling.py (e.g. --override or --global-calls).
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args, extra = parser.parse_known_args()
    build, output = args.build_dir.resolve(), args.output.resolve()
    cache = dict(line.split('=', 1) for line in (build / 'CMakeCache.txt').read_text().splitlines()
                 if '=' in line and not line.startswith(('//', '#')))
    mode = cache.get('GUNDAM_GRAPH_UNROLLED_SEARCH:BOOL')
    if mode not in ('ON', 'OFF'):
        parser.error('Reconfigure this build with -DGUNDAM_GRAPH_UNROLLED_SEARCH=OFF or ON')
    if cache.get('CMAKE_BUILD_TYPE:STRING', '').upper() != 'RELEASE':
        parser.error('Use a Release build for this benchmark')
    if any(flag.split('=')[0] in ('--worker', '--output', '--work-dir') for flag in extra):
        parser.error('Do not override internal worker/output/work-dir options')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(build / 'src/PythonInterface') + os.pathsep + env.get('PYTHONPATH', '')
    # Fail before loading the production dataset if an installed module/library wins.
    probe = subprocess.check_output([sys.executable, '-c',
        'import GUNDAM, json; from pathlib import Path; '
        'p=Path("/proc/self/maps"); '
        'print(json.dumps({"module": GUNDAM.__file__, "libraries": sorted(set('
        'line.split()[-1] for line in p.read_text().splitlines() '
        'if "/" in line and "gundam" in line.split()[-1].lower())) if p.exists() else []}))'],
        env=env, text=True)
    loaded = json.loads(probe.splitlines()[-1])
    for path in [loaded['module'], *loaded['libraries']]:
        if build not in Path(path).resolve().parents:
            parser.error(f'Library outside selected build: {path}; check LD_LIBRARY_PATH')
    libraries = {str(path.relative_to(build)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in build.rglob('*') if path.is_file()
                 and path.suffix in ('.so', '.dylib') and 'gundam' in path.name.lower()}
    provenance = {'graphUnrolledSearch': mode, 'buildDirectory': str(build),
                  'cmakeCache': cache, 'loaded': loaded, 'librarySha256': libraries}
    command = [sys.executable, str(HERE / '229CudaBackendProfiling.py'),
               '--work-dir', str(args.work_dir.resolve()), '--config', 'configOA2024.yaml',
               '--override', 'override/v13Prod.yaml', 'override/backendCudaTest.yaml',
               '--cases', 'CUDA-cachedResponses-interleaved', 'CUDA-recomputeResponses-interleaved',
               '--execution-mode', 'resident', '--threads', '4', '--calls', '100',
               '--global-calls', '100', '--trials', '3', '--warmup', '10',
               '--scenarios', 'constant', 'local1', 'global', *extra, '--output', str(output)]
    print(f'Graph unrolled search: {mode}; module {loaded["module"]}', flush=True)
    subprocess.run(command, env=env, check=True)
    (output / 'graph-search-build.json').write_text(json.dumps(provenance, indent=2) + '\n')
    settings = read_json(output / 'settings.json')
    for case in settings['cases']:
        if case == 'CPU':
            continue
        model = read_json(output / case / 'model.json')
        if model.get('graphUnrolledSearch') != (mode == 'ON'):
            raise RuntimeError('Backend binary does not match the CMake option; rebuild before running')
        print(f'{case}: {model["graphDials"]} graph dials', flush=True)
        if not model['graphDials']:
            raise RuntimeError('No graph dials in this model; cannot benchmark graph search')
    if args.reference is None:
        return
    reference = args.reference.resolve()
    before = read_json(reference / 'graph-search-build.json')
    if before['graphUnrolledSearch'] != 'OFF' or mode != 'ON':
        raise RuntimeError('Use the OFF run as reference for the ON run')
    old_settings = read_json(reference / 'settings.json')
    for key in ('cases', 'scenarios', 'trials', 'calls', 'global_calls', 'warmup', 'seed',
                'step_scale', 'threads', 'parameter_match', 'execution_mode', 'event_variables'):
        if settings[key] != old_settings[key]:
            raise RuntimeError(f'Benchmark settings differ: {key}')
    spec = importlib.util.spec_from_file_location('profiling', HERE / '229CudaBackendProfiling.py')
    profiling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiling)
    lines = ['# Graph search: loop / unrolled speedup', '',
             '| Case | Scenario | Evaluation | Dial responses |', '|---|---|---:|---:|']
    for case in settings['cases']:
        if read_json(output / case / 'resolved-fitter-config.json') != read_json(reference / case / 'resolved-fitter-config.json'):
            raise RuntimeError(f'Resolved configurations differ: {case}')
        profiling.compare(output, [case], settings['scenarios'], settings['trials'],
                          settings['relative_tolerance'], settings['absolute_tolerance'],
                          reference_directory=reference / case,
                          comparison_name=f'graph-search-comparison-{case}.json')
        left, right = read_json(reference / case / 'summary.json'), read_json(output / case / 'summary.json')
        for scenario in settings['scenarios']:
            ratios = []
            for field in ('evaluationSeconds', 'cachedDialStageSeconds'):
                a = left['scenarios'][scenario]['timings'][field]['medianMs']
                b = right['scenarios'][scenario]['timings'][field]['medianMs']
                ratios.append(f'{a / b:.3f}x' if b else 'n/a')
            lines.append('| ' + ' | '.join([case, scenario, *ratios]) + ' |')
    lines += ['', 'Above 1 means unrolled is faster. Initialization is excluded.',
              'Constant cached calls may skip graph evaluation. Inspect model.json/native.log',
              'for the graph population; full response timings also include other dial types.',
              'Numerical checks: every measured likelihood and final trial histogram/errors.']
    report = '\n'.join(lines) + '\n'
    (output / 'graph-search-comparison.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
