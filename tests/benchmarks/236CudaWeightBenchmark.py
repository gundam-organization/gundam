#!/usr/bin/env python3
"""Resident standard-dial/weight benchmark and targeted Nsight Compute capture.

All remaining arguments are forwarded to 229CudaBackendProfiling.py. Each run
loads the dataset once and activates subsequent backends from loaded events.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--mode', choices=['benchmark', 'ncu', 'preflight'], default='benchmark')
    parser.add_argument('--python-module-dir', type=Path)
    parser.add_argument('--test-executable', type=Path, help='Run CUDA packing/propagation tests before loading inputs')
    parser.add_argument('--ncu', default='ncu')
    parser.add_argument('--ncu-set', default='full')
    parser.add_argument('--cache-control', choices=['all', 'none'], default='all')
    parser.add_argument('--output', type=Path,
                        default=Path('benchmark-236CudaWeightBenchmark-' + time.strftime('%Y%m%d%H%M%S')))
    own, forwarded = parser.parse_known_args(argv)
    environment = os.environ.copy()
    if own.python_module_dir:
        module = own.python_module_dir.resolve()
        if not module.is_dir():
            parser.error(f'Module directory does not exist: {module}')
        environment['PYTHONPATH'] = str(module) + (os.pathsep + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    profile = Path(__file__).with_name('229CudaBackendProfiling.py')
    if '-h' in forwarded or '--help' in forwarded:
        parser.print_help()
        subprocess.run([sys.executable, str(profile), '--help'], check=True)
        return
    if own.mode == 'preflight':
        output = own.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        results = {}
        for name, command in [('ncu', [own.ncu, '--version']), ('gpu', ['nvidia-smi']),
                              ('nvcc', ['nvcc', '--version'])]:
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                results[name] = dict(command=command, returncode=result.returncode,
                                     stdout=result.stdout, stderr=result.stderr)
            except (OSError, subprocess.TimeoutExpired) as error:
                results[name] = dict(command=command, error=str(error))
        results['environment'] = {key: environment.get(key) for key in
            ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_ID', 'PYTHONPATH', 'LD_LIBRARY_PATH')}
        (output/'preflight.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps(results, indent=2))
        return
    if own.test_executable:
        executable = own.test_executable.resolve()
        if not executable.is_file():
            parser.error(f'CUDA test executable does not exist: {executable}')
        subprocess.run([str(executable), '--gtest_filter=CudaCompactDialModel.*:CudaBackendPropagation.*'],
                       env=environment, check=True)
    if '--execution-mode' in forwarded or '--worker' in forwarded:
        parser.error('This entry point manages resident execution; do not pass --execution-mode or --worker')
    defaults = ['--execution-mode', 'resident', '--require-device-likelihood', '--require-v100',
                '--require-weight-inventory',
                '--threads', '4', '--calls', '100', '--global-calls', '40', '--trials', '3', '--warmup', '8']
    if own.mode == 'benchmark':
        defaults += ['--cases', 'CPU', 'CUDA-cachedResponses-interleaved',
                     'CUDA-cachedResponses-interleaved-compact', 'CUDA-recomputeResponses-interleaved',
                     'CUDA-recomputeResponses-interleaved-compact']
        command = [sys.executable, '-u', str(profile), *defaults, *forwarded, '--output', str(own.output.resolve())]
    else:
        if not shutil.which(own.ncu):
            parser.error('Nsight Compute is unavailable; use --ncu /path/to/ncu')
        output = own.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        defaults += ['--cases', 'CUDA-cachedResponses-interleaved',
                     'CUDA-cachedResponses-interleaved-compact', '--scenarios', 'local1', 'global',
                     '--calls', '1', '--global-calls', '1', '--trials', '1', '--profile-capture']
        command = [own.ncu, '--target-processes', 'all', '--profile-from-start', 'off',
                   '--replay-mode', 'kernel', '--set', own.ncu_set, '--clock-control', 'none',
                   '--cache-control', own.cache_control, '--kernel-name-base', 'function',
                   '--kernel-name', 'regex:evaluateResponses|evaluateSelectedResponses|reweightEvents',
                   '--export', str(output/'weights'),
                   sys.executable, '-u', str(profile), *defaults, *forwarded,
                   '--output', str(output/'capture')]
        (output/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    print('Launching:', ' '.join(command), flush=True)
    subprocess.run(command, env=environment, check=True)
    if own.mode == 'ncu' and not (output/'weights.ncu-rep').is_file():
        raise RuntimeError('No Nsight report produced; inspect GPU support, counter permissions and kernel filters')


if __name__ == '__main__':
    main()
