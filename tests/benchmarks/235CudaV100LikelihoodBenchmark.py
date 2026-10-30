#!/usr/bin/env python3
"""Validate and benchmark resident CPU/CUDA statistical likelihoods on a V100.

Run inside a GPU allocation using a CUDA-enabled GUNDAM Python module. No fit,
Hesse or input reduction. Independent workers support dynamic/smoothed inputs.
The automatic CUDA launch policy is a hypothesis, not a measured V100 optimum.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location('profiling', Path(__file__).with_name('229CudaBackendProfiling.py'))
profiling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profiling)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--override', nargs='*', default=[])
    p.add_argument('--python-module-dir', type=Path, help='Directory containing the matching CUDA-enabled GUNDAM module')
    p.add_argument('--test-executable', type=Path, help='Optional gundamGTest_cudaBackend.exe: run device/likelihood tests first')
    p.add_argument('--block-sizes', nargs='+', choices=['auto', '32', '64', '128', '256'], default=['auto', '256'])
    p.add_argument('--reuse-modes', nargs='+', choices=['on', 'off'], default=['on'])
    p.add_argument('--likelihoods', nargs='+', choices=['configured', 'PoissonLLH', 'BarlowLLH_BANFF_OA2021'], default=['configured'])
    p.add_argument('--cases', nargs='+', choices=profiling.CASES,
                   default=['CPU', 'CUDA-cachedResponses-interleaved', 'CUDA-recomputeResponses-interleaved'])
    p.add_argument('--execution-mode', choices=['isolated', 'resident'], default='isolated',
                   help='Isolated supports observable/binning responses; resident loads fixed-bin inputs once per experiment')
    p.add_argument('--scenarios', nargs='+', choices=['constant','local1','local2','global'],
                   default=['constant','local1','local2','global'])
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device-index', type=int, default=0)
    p.add_argument('--calls', type=int, default=100)
    p.add_argument('--global-calls', type=int, default=40)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--warmup', type=int, default=8)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--step-scale', type=float, default=.1)
    p.add_argument('--parameter-match')
    p.add_argument('--event-variables', nargs='*', default=None)
    p.add_argument('--require-rebinning', action='store_true')
    p.add_argument('--allow-other-gpu', action='store_true', help='Disable the selected-device V100 name check')
    p.add_argument('--relative-tolerance', type=float, default=1e-9)
    p.add_argument('--absolute-tolerance', type=float, default=1e-7)
    p.add_argument('--output', type=Path, default=Path('benchmark-235CudaV100LikelihoodBenchmark-'+time.strftime('%Y%m%d%H%M%S')))
    args = p.parse_args(argv)
    for name in ('block_sizes', 'reuse_modes', 'likelihoods', 'cases', 'scenarios'):
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    args.work_dir = args.work_dir.resolve()
    args.config = (args.work_dir / args.config).resolve()
    args.output = args.output.resolve()
    if not args.config.is_file():
        p.error('Configuration does not exist: '+str(args.config))
    if min(args.calls, args.global_calls, args.trials, args.threads) <= 0 or min(args.warmup, args.device_index) < 0:
        p.error('Counts must be positive; warmup and device-index may be zero')
    if args.require_rebinning and args.execution_mode != 'isolated':
        p.error('--require-rebinning requires isolated execution')
    return args


def command(args, probability, block, reuse, output):
    result = [sys.executable, '-u', str(Path(__file__).with_name('229CudaBackendProfiling.py')),
              '--work-dir', str(args.work_dir), '--config', str(args.config),
              '--execution-mode', args.execution_mode, '--cases', *args.cases,
              '--scenarios', *args.scenarios,
              '--likelihood', probability, '--likelihood-block-size', '0' if block == 'auto' else block,
              '--require-device-likelihood', '--output', str(output)]
    for name in ('threads', 'device_index', 'calls', 'global_calls', 'trials', 'warmup', 'seed',
                 'step_scale', 'relative_tolerance', 'absolute_tolerance'):
        result.extend(['--'+name.replace('_', '-'), str(getattr(args, name))])
    if args.override:
        result.extend(['--override', *args.override])
    if reuse == 'off':
        result.append('--disable-prediction-reuse')
    if not args.allow_other_gpu:
        result.append('--require-v100')
    if args.parameter_match:
        result.extend(['--parameter-match', args.parameter_match])
    if args.event_variables is not None:
        result.extend(['--event-variables', *args.event_variables])
    if args.require_rebinning:
        result.append('--require-rebinning')
    return result


def report(args, summary):
    lines = ['# V100 statistical likelihood benchmark', '',
             'Seeded evaluations; no full minimization/Hesse. Medians in milliseconds.',
             'GPU stage times overlap wall times; do not add these columns.', '',
             '| Likelihood | Block | Reuse | Case | Scenario | Evaluation | GPU likelihood | LLH readback bytes |',
             '|---|---|---|---|---|---:|---:|---:|']
    for entry in summary['experiments']:
        data = json.loads((args.output/entry['directory']/'summary.json').read_text())
        for case, case_data in data['cases'].items():
            for scenario, measurement in case_data['scenarios'].items():
                times = measurement['timings']
                lines.append(f"| {entry['likelihood']} | {entry['block']} | {entry['reuse']} | {case} | {scenario} | "
                             f"{times['evaluationSeconds']['medianMs']:.4f} | "
                             f"{times['likelihoodDeviceSeconds']['medianMs']:.4f} | "
                             f"{measurement['counts']['likelihoodReadbackBytes']['max']} |")
    lines.extend(['', 'Every experiment checks CPU/CUDA outputs when CPU is selected.',
                  'Cross-experiment comparisons also reject changes in replay points, likelihoods or histograms.',
                  'Block-size and reuse sweeps are all the optimized parallel reduction, not the historical serial kernel.',
                  'Auto uses 64-thread blocks through 8192 model bins, otherwise 256. Treat this as a starting hypothesis.',
                  'Compare full evaluation medians and trial distributions before choosing a setting.'])
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    if args.python_module_dir:
        module = args.python_module_dir.resolve()
        if not module.is_dir():
            raise RuntimeError('Python module directory does not exist: '+str(module))
        env['PYTHONPATH'] = str(module)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    settings = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    profiling.write_json(args.output/'settings.json', settings)
    summary = {'complete': False, 'experiments': []}
    profiling.write_json(args.output/'summary.json', summary)
    if args.test_executable:
        with (args.output/'device-tests.log').open('w') as log:
            subprocess.run([str(args.test_executable.resolve()),
                '--gtest_filter=BackendLikelihood.*:CudaBackendDevice.*'],
                stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    references = {}
    for probability in args.likelihoods:
        for block in args.block_sizes:
            for reuse in args.reuse_modes:
                name = f'{probability}-block{block}-reuse{reuse}'
                directory = args.output/name
                profiling.progress('V100 matrix: '+name)
                subprocess.run(command(args, probability, block, reuse, directory), env=env, check=True)
                if probability not in references:
                    reference_case = 'CPU' if 'CPU' in args.cases else args.cases[0]
                    references[probability] = directory/reference_case
                else:
                    profiling.compare(directory, args.cases, args.scenarios,
                        args.trials, args.relative_tolerance, args.absolute_tolerance,
                        reference_directory=references[probability], comparison_name='matrix-comparison.json')
                summary['experiments'].append({'likelihood': probability, 'block': block,
                    'reuse': reuse, 'directory': name})
                profiling.write_json(args.output/'summary.json', summary)
                report(args, summary)
    summary['complete'] = True
    profiling.write_json(args.output/'summary.json', summary)
    profiling.progress('SUCCESS: '+str(args.output))


if __name__ == '__main__':
    main()
