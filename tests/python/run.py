#!/usr/bin/env python3
"""Run Python tests with this interpreter, independently of gundam-tests.sh."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['fast', 'regular', 'benchmarks'], default='fast')
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
    for script in scripts:
        work = output / script.stem
        work.mkdir()
        command = [sys.executable, '-B', *(['-u'] if args.verbose else []), str(script), *arguments]
        with (work / 'test.log').open('wb') as log:
            if args.verbose:
                print('RUN:', script.name, flush=True)
                with subprocess.Popen(command, cwd=work, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT) as result:
                    while chunk := result.stdout.read1(8192):
                        log.write(chunk)
                        log.flush()
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
            else:
                result = subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append(script.name)
            print('FAIL:', script.name, flush=True)
            if not args.verbose:
                print((work / 'test.log').read_text(errors='replace'), flush=True)
        else:
            print('PASS:', script.name, flush=True)
    print(f'{len(scripts) - len(failures)}/{len(scripts)} scripts passed; logs: {output}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
