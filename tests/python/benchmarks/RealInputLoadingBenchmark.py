#!/usr/bin/env python3
"""Time real input loading, retaining logs and per-dataset phase measurements.

Run outside run.py, for example:
  python RealInputLoadingBenchmark.py --executable /path/to/gundamFitter \
    --workdir /path/to/fd-2024 --config config/fdOa2024.yaml \
    --override config/overrides/postfitND.yaml --threads 1 2 4 --repeat 2 \
    --output /tmp/loading-benchmark

The environment is inherited. Runs are sequential, do not flush the OS file
cache, and write all fitter outputs beneath the requested output directory.
With --loading-only, timing ends at the prior propagation log message and the
fitter is terminated; its ROOT output file is not intended for analysis.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def run(args, threads, repetition):
    directory = args.output / f't{threads}-r{repetition}'
    directory.mkdir()
    command = [str(args.executable), '-c', args.config,
               '-d', '-a', '-t', str(threads), '--out-dir', str(directory)]
    if args.override:
        command += ['-of', *args.override]
    if args.fraction is not None:
        command += ['-fe', str(args.fraction)]
    result = dict(command=command, threads=threads, repetition=repetition,
                  fraction=args.fraction, loading_only=args.loading_only, datasets=[])
    dataset = None
    phase_starts = {}
    start = time.perf_counter()
    with (directory / 'fitter.log').open('w') as log:
        with subprocess.Popen(command, cwd=args.workdir, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors='replace') as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    now = time.perf_counter() - start
                    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line).strip()
                    if args.loading_only and 'Propagating prior parameters on events...' in text:
                        result['loading_complete_seconds'] = now
                        process.terminate()
                        # Drain remaining output so the child cannot block on
                        # its pipe during cleanup; fitter output is disposable.
                        try:
                            remaining, _ = process.communicate(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            remaining, _ = process.communicate()
                        log.write(remaining)
                        break
                    if 'Loading dataset: ' in text:
                        dataset = dict(name=text.split('Loading dataset: ', 1)[1], start=now)
                        result['datasets'].append(dataset)
                        phase_starts = {}
                    if dataset is None:
                        continue
                    match = re.search(r'Will read (\d+) event entries', text)
                    if match:
                        dataset['input_entries'] = int(match[1])
                        phase_starts['selection'] = now
                    for marker, phase in [('Merging thread results...', 'selection'),
                                          ('Releasing ROOT readers and loading buffers...', 'fill')]:
                        if marker in text and phase in phase_starts:
                            dataset[phase + '_seconds'] = now - phase_starts.pop(phase)
                            print(f"t={threads} {dataset['name']} {phase}: "
                                  f"{dataset[phase + '_seconds']:.3f}s", flush=True)
                    if 'Reading dataset and loading...' in text:
                        phase_starts['fill'] = now
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            result['returncode'] = process.wait()
    result['wall_seconds'] = time.perf_counter() - start
    (directory / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"t={threads} run={repetition}: {result['wall_seconds']:.3f}s "
          f"exit={result['returncode']}", flush=True)
    if result['returncode'] and 'loading_complete_seconds' not in result:
        raise RuntimeError(f'Fitter failed; see {directory / "fitter.log"}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--workdir', required=True, type=Path)
    parser.add_argument('--config', required=True)
    parser.add_argument('--override', action='append', default=[])
    parser.add_argument('--threads', nargs='+', type=int, default=[4])
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--fraction', type=float)
    parser.add_argument('--loading-only', action='store_true',
                        help='End timing at the start of prior propagation, then terminate the fitter')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.executable = args.executable.resolve(strict=True)
    args.workdir = args.workdir.resolve(strict=True)
    args.output = args.output.resolve()
    if args.repeat < 1 or any(threads < 1 for threads in args.threads):
        parser.error('repeat and thread counts must be positive')
    if args.fraction is not None and not 0 < args.fraction <= 1:
        parser.error('fraction must be in (0, 1]')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for repetition in range(1, args.repeat + 1):
        for threads in args.threads:
            results.append(run(args, threads, repetition))
            (args.output / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
