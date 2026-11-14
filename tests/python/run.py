#!/usr/bin/env python3
"""Run Python tests with this interpreter, independently of gundam-tests.sh."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'fast'))
from shared.BackendTestSupport import mps_available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['fast', 'regular', 'benchmarks'], default='fast')
    parser.add_argument('--require-backend', choices=['CPU', 'MPS', 'CUDA'])
    parser.add_argument('--cuda-runner', type=Path, help='Built CUDA device-checking launcher')
    parser.add_argument('--test', help='Run one script by filename in the selected suite')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Stream test output to the terminal while keeping log files')
    parser.add_argument('--list', action='store_true', help='List scripts without running them')
    parser.add_argument('--output', type=Path, help='New directory for logs and generated files')
    parser.add_argument('arguments', nargs=argparse.REMAINDER, help='Arguments forwarded after -- (requires --test)')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent / args.suite
    scripts = sorted(path for path in directory.glob('*.py')
                     if path.name[0].isdigit() or path.name.startswith('test'))
    if args.test:
        scripts = [path for path in scripts if path.name == args.test]
        if not scripts:
            parser.error('Test not found in ' + str(directory) + ': ' + args.test)
    if args.arguments and not args.test:
        parser.error('Forwarding arguments requires --test')
    if args.list:
        for script in scripts:
            print(script.relative_to(directory.parent))
        return 0

    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix='gundam-python-' + args.suite + '-'))
    print('Python:', sys.executable, flush=True)
    print('Output:', output, flush=True)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    failures = []
    skipped = []
    for script in scripts:
        work = output / script.stem
        work.mkdir()
        backend = args.require_backend
        if backend is None and script.name.startswith('74'):
            backend = 'MPS'
        if backend is None and script.name[:3] in ('760', '761', '762', '766'):
            backend = 'CUDA'
        reason = None
        if script.name.startswith('7'):
            import GUNDAM
            if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
                reason = 'backend support was not compiled'
        if backend == 'MPS' and not mps_available():
            reason = 'no usable Metal device'
        if reason:
            skipped.append(script.name)
            (work / 'test.log').write_text('SKIP: ' + reason + '\n')
            print('SKIP:', script.name, '-', reason, flush=True)
            continue
        environment = os.environ.copy()
        if backend == 'CUDA':
            environment['GUNDAM_TEST_CUDA'] = '1'
            if not args.cuda_runner:
                parser.error('CUDA device tests require --cuda-runner from a CUDA build')
        command = [sys.executable, '-B', *(['-u'] if args.verbose else []), str(script), *arguments]
        if backend == 'CUDA':
            command.insert(0, str(args.cuda_runner))
        with (work / 'test.log').open('wb') as log:
            if args.verbose:
                print('RUN:', script.name, flush=True)
                with subprocess.Popen(command, cwd=work, env=environment, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT) as result:
                    while chunk := result.stdout.read1(8192):
                        log.write(chunk)
                        log.flush()
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
            else:
                result = subprocess.run(command, cwd=work, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 77:
            skipped.append(script.name)
            print('SKIP:', script.name, flush=True)
            if not args.verbose:
                print((work / 'test.log').read_text(errors='replace'), flush=True)
        elif result.returncode:
            failures.append(script.name)
            print('FAIL:', script.name, flush=True)
            if not args.verbose:
                print((work / 'test.log').read_text(errors='replace'), flush=True)
        else:
            print('PASS:', script.name, flush=True)
    print(f'{len(scripts) - len(failures) - len(skipped)}/{len(scripts)} scripts passed; {len(skipped)} skipped; logs: {output}')
    return 1 if failures else (77 if skipped and len(skipped) == len(scripts) else 0)


if __name__ == '__main__':
    sys.exit(main())
