#!/usr/bin/env python3
"""Summarize GUNDAM CPU JSONL profiles with the Python standard library."""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def milliseconds(value):
    return 'unavailable' if value is None else f'{value * 1000:.3f} ms'


def summarize(path, include_startup=False):
    runs = defaultdict(list)
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A killed job may leave an incomplete final record.
            if index == len(lines) - 1:
                print(f'Warning: ignoring incomplete last line in {path}', file=sys.stderr)
                continue
            raise
        if record.get('schemaVersion') != 1:
            raise ValueError(f'Unsupported profile schema: {record.get("schemaVersion")}')
        runs[record['runId']].append(record)
    for run, records in runs.items():
        metadata = next(r for r in records if r['kind'] == 'metadata')
        all_samples = [r for r in records if r['kind'] == 'sample']
        samples = [r for r in all_samples if include_startup or r['call'] > 5]
        print(f'\n{path} / {run}')
        print(f'CPU threads={metadata["threads"]}; host={metadata["system"].get("host", "unknown")}; '
              f'build={metadata["buildType"]}; compiler optimization={metadata["compilerOptimizationEnabled"]}')
        print(f'Events={metadata["events"]:,}; unique dials={metadata["uniqueDials"]:,}; '
              f'event/dial references={metadata["eventDialReferences"]:,}')
        print(f'Samples={len(samples)} ({len(all_samples)} including startup); '
              f'commit={metadata["commit"]}; compiler={metadata["compiler"]}')
        if 'weightScheduling' in metadata:
            details = (f'block size={metadata["weightBlockSize"]} events'
                       if 'weightBlockSize' in metadata else f'work blocks={metadata["weightBlockCount"]}')
            print(f'Weight scheduling={metadata["weightScheduling"]} '
                  f'(requested {metadata["weightSchedulingRequested"]}); {details}')
        allowed = metadata['system'].get('allowedCpus')
        if allowed is not None:
            print(f'Allowed logical CPUs ({len(allowed)}): {allowed}')
        if not samples:
            print('No samples after startup. Use --include-startup or run more propagations.')
            continue
        print('Median backend stage durations:')
        for stage in ('cache', 'weights', 'histograms', 'variables', 'binning', 'likelihood'):
            values = [s['backendTimingSeconds'][stage] for s in samples]
            print(f'  {stage:12s} median {milliseconds(median(values))}, '
                  f'p95 {milliseconds(percentile(values, .95))}, max {milliseconds(max(values))}')
        print(f'  cache prep   {milliseconds(median(s["cachePreparationSeconds"] for s in samples))}')
        print(f'Median changed parameters={median(len(s["changedParameters"]) for s in samples):g}; '
              f'dirty cached dials={median(s["dirtyCachedDials"] for s in samples):g}')
        groups = defaultdict(list)
        for sample in samples:
            groups[len(sample['changedParameters'])].append(sample['backendTimingSeconds']['cache'])
        print('Cache duration by number of changed parameters:')
        for count, values in sorted(groups.items()):
            print(f'  {count:4d} changed: {len(values):4d} samples, median {milliseconds(median(values))}')
        for stage_index, name in enumerate(('cache', 'weights', 'histograms')):
            stage_samples = [s['stages'][stage_index] for s in samples]
            tails = [s['joinTailSeconds'] for s in stage_samples]
            print(f'{name} workers (join tail median {milliseconds(median(tails))}):')
            for thread in range(metadata['threads']):
                workers = [w for s in stage_samples for w in s['workers'] if w['thread'] == thread]
                if not workers:
                    continue
                cpus = sorted({w[key] for w in workers for key in ('cpuAtStart', 'cpuAtEnd') if w[key] is not None})
                print(f'  {thread:2d}: work {milliseconds(median(w["wallSeconds"] for w in workers))}, '
                      f'CPU {milliseconds(median(w["cpuSeconds"] for w in workers))}, CPUs={cpus or "unavailable"}'
                      + (f'; median events={median(w["weightEvents"] for w in workers):g}, '
                         f'blocks={median(w["weightBlocks"] for w in workers):g}'
                         if name == 'weights' and 'weightEvents' in workers[0] else ''))
            if name == 'cache':
                evaluated = [sum(w['evaluatedDials'] for w in s['workers']) for s in stage_samples]
                skipped = [sum(w['unchangedInputs'] for w in s['workers']) for s in stage_samples]
                print(f'  Median evaluated dials={median(evaluated):g}; unchanged transformed inputs={median(skipped):g}')
        print('Slowest sampled cache calls:')
        for sample in sorted(samples, key=lambda s: s['backendTimingSeconds']['cache'], reverse=True)[:5]:
            print(f'  call {sample["call"]}: {milliseconds(sample["backendTimingSeconds"]["cache"])}, '
                  f'{sample["dirtyCachedDials"]:,} candidate dials; '
                  f'changed parameter indices={sample["changedParameters"][:12]}'
                  + (' ...' if len(sample['changedParameters']) > 12 else ''))
        print('Process CPU/wall median: '

              + str(median(s['process']['cpuSeconds'] / s['process']['wallSeconds']
                           for s in samples if s['process']['cpuSeconds'] is not None and s['process']['wallSeconds'] > 0)))
        print('Raw files retain CPU topology, start/end offsets, context switches, faults, NUMA and cgroup data.')
        print('These probes do not measure hardware cache misses or DRAM bandwidth.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profiles', type=Path, nargs='+')
    parser.add_argument('--include-startup', action='store_true')
    args = parser.parse_args()
    for path in args.profiles:
        summarize(path, args.include_startup)


if __name__ == '__main__':
    main()
