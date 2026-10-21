# Performance benchmarks

These scripts measure performance; they are not part of the regular regression
suite. Use a Release build with the Python interface enabled and select the
matching installed `GUNDAM` Python module.

## Single-thread propagation and function sampling

```sh
python3 "$REPO_DIR/gundam/tests/benchmarks/224CpuSingleThreadBenchmark.py" \
  --config configOA2024.yaml --hardware-counters --perf-record \
  --override override/v13Prod.yaml override/backendCpuTest.yaml
```

This sets the complete GUNDAM thread pool to one: weights, cache workers and
histogram filling use one thread. This differs from the scaling benchmark's
one-active-weight-worker case, which retains a larger histogram pool. Dataset
loading also uses the single-thread setting and may take longer. The input
selection is unchanged and no event-count limit is added. The complete production
model still needs its usual memory allocation, even with only one compute thread.

Default work is 40 constant-point timed evaluations, five separate counter probes,
and (when `--perf-record` can attach) 40 separately sampled evaluations, plus
warmup/priming. All response caches are primed; this isolates steady-state weight
and histogram costs, not the cost of updating all spline responses. There is no
Migrad/HESSE run and no dataset reload between passes.

Output under `cpu-single-thread-<timestamp>-<PID>/` includes the normal JSON
reports with cache, weight, histogram, statistical likelihood, variable, binning,
host input-preparation and complete-evaluation times. Their sum can differ from
complete time because wrapper/penalty work is outside the listed timers. Select
the rebuilt Python interface to expose the statistical likelihood timer.

Optional Linux `perf` sampling attaches only after model loading and ordinary
measurements. It samples the complete repeated likelihood evaluation (including
Python wrapper work), not just weights. It uses `cycles:u` at 199 Hz and DWARF
call stacks, then writes `perf.data`, `perf-record.log`, `perf-report.txt` and
`perf-status.json`. An additional `perf-annotate.txt` shows sampled assembly
for functions above 1% overhead when disassembly is available. The report ranks
self overhead by library/function; call
stacks remain in `perf.data`. The short attachment delay may leave a small amount
of startup/idle sampling outside the evaluation loop. Sampling results are never
mixed into ordinary timing trials or the phase-scoped counter probes.

An explicit `--perf-record` request first runs a tiny perf permission/tool check,
before dataset loading. If perf is absent or refused, the script stops immediately
and writes `perf-status.json` / `perf-preflight.log` as available. Omit the flag
when only stage/counter measurements are wanted. If later sampling/reporting fails,
its status/log and already completed timing results remain available. No
installation or privilege escalation is attempted. Function/line resolution depends on the
installed binary's symbols: Release can provide function attribution when symbols
are retained; RelWithDebInfo is preferable for source/instruction analysis. A
sampled instruction near a cache miss is not by itself proof of the stall's cause.
See the [perf record](https://man7.org/linux/man-pages/man1/perf-record.1.html) and
[perf report](https://man7.org/linux/man-pages/man1/perf-report.1.html) manuals.

## Resident CPU weight scaling

On an allocation with eight CPUs, run from the production configuration directory:

```sh
python3 "$REPO_DIR/gundam/tests/benchmarks/223CpuWeightScalingBenchmark.py" \
  --config configOA2024.yaml --threads 8 --workers 1 2 4 8 \
  --calls 30 --trials 1 --rounds 2 --probe-calls 5 \
  --override override/v13Prod.yaml override/backendCpuTest.yaml
```

The dataset is loaded once using the configured input files, selections and
limits. The benchmark does not add an event-count limit. It evaluates the same
initialized Asimov point with primed dial responses. All event weights are still
recomputed. Automatic copies into host event/histogram objects are disabled as
in the fit loop; the likelihood is checked on every timed call and the final
histogram is published outside timing for an exact comparison.

Only the number of **active weight workers** changes. The resident pool remains
at eight threads, with the same affinity, allocation and memory placement; the
cache and histogram stages retain their original workers. Inactive weight
workers return immediately but still participate in the pool dispatch/join.
This measures the weight stage's scaling within one resident allocation, not
the complete fit's scaling across separate one/two/four/eight-CPU jobs. It does
not pin individual workers or change NUMA policy.

Defaults perform 240 timed evaluations and 40 separate probe evaluations, plus
warmup/priming. The second round reverses worker counts to expose order effects.
No minimization, HESSE, global parameter updates or repeated dataset loading is
required. Dataset initialization can still take a long time; progress heartbeats
appear every 30 seconds and full native output is saved to `native.log`.

Results are written under `cpu-weight-scaling-<timestamp>-<PID>/`:

- `settings.json` and `model.json`: invocation, model size, pool layout, affinity,
  host and SLURM allocation information.
- `summary.json`: stage median/mean/p95 times, weight speedup relative to one
  active worker, and dial-reference throughput (not measured memory bandwidth).
- `round*-w*.json`: raw timing trials and separate per-worker probes, including
  wall time, events, dial references and claimed blocks. Worker clocks cover the
  calculation, excluding the pool's final synchronization. The stage time also
  includes dispatch/join overhead. Probe timings include instrumentation overhead
  and are not mixed with the main timing trials.

Completed cases and trials are checkpointed. Each completed trial leaves a
`.partial.json` until its case is complete; completed cases and the summary
survive interruption. Return the whole output directory.

Unequal worker durations together with unequal reference counts suggest load
imbalance. A throughput plateau with balanced workers suggests a shared resource
limit, but does not prove memory-bandwidth saturation. Hardware counters, if
available on the worker node, are needed to distinguish cache misses, memory
stalls and compute limits. No `perf` dependency or privilege is required here.

## Phase-scoped Linux hardware probes

After rebuilding/installing GUNDAM, add `--hardware-counters` to the scaling
benchmark. A short comparison of one and eight active workers is sufficient to
start:

```sh
python3 "$REPO_DIR/gundam/tests/benchmarks/223CpuWeightScalingBenchmark.py" \
  --config configOA2024.yaml --threads 8 --workers 1 8 \
  --calls 20 --rounds 2 --probe-calls 5 --hardware-counters \
  --override override/v13Prod.yaml override/backendCpuTest.yaml
```

A quick access check is printed and saved to `counter-preflight.json` before
loading the dataset; per-worker access is checked again in the actual probes.
Denied counters do not prevent collecting OS/thread timing data.

This loads the dataset once, runs 80 ordinary timed evaluations and 20 recorded
probe evaluations, plus warmup/priming and one counter-initialization evaluation
per case. Counters are off for all ordinary timing trials. Linux worker threads
open their own counters through `perf_event_open`, without an external `perf`
process, system-wide monitoring, new library or elevated privilege request.
Only userspace execution of the weight-job wrapper is counted; dial response
updates, histograms, host publication and the final pool join are excluded.
Opening counters, exporting reports and reading counters are outside the worker
wall-time measurement. Hardware boundaries include a small amount of probe/setup
code around the kernel, so these are diagnostic samples, not replacement timings.

The optional probe requests cycles, retired instructions, generic cache references
and misses, LLC read misses and data-TLB read misses. Every event is independent:
one unsupported/denied event does not disable the others. Reports include errno,
raw deltas, enabled/running time deltas and multiplex-scaled estimates. Counters
remain disabled between weight jobs and are closed after each case. Counter and
snapshot storage scales with worker count, not event count.

The per-worker `weightProbe` record also includes thread CPU time, wall time,
CPU IDs at both boundaries, voluntary/involuntary context switches and page
faults. A matching CPU at both ends does not exclude intervening migrations.
Page faults describe virtual-memory paging, not CPU cache misses. Even when
hardware access is refused, the Linux thread/OS measurements are retained.
On non-Linux platforms hardware/OS fields are explicitly unavailable; portable
weight timings and correctness comparisons still run.

`summary.json` adds counter coverage, the minimum PMU running fraction, an IPC
estimate, instructions per dial reference, generic cache misses per 1,000 dial
references and median worker CPU/wall ratio. Raw data remain in `round*-w*.json`.
No missing counter is treated as zero. Low running fractions weaken estimates:
independent events may have sampled different execution intervals. Generic cache
event mappings depend on the CPU/kernel; counts must not be converted directly
into RAM bytes or interpreted as a universal cache-miss rate.

The questions are whether instructions per reference stay stable while cycles
increase, whether cache/TLB misses accompany that change, and whether workers
lose CPU time to scheduling. Low IPC alone cannot distinguish memory latency,
SMT contention and multiplication dependencies. Cache misses alone do not prove
RAM bandwidth saturation. Results guide the next experiment rather than applying
a hardware-dependent threshold or changing the fit policy automatically.

Counter semantics and Linux access restrictions are described in the
[Linux `perf_event_open` manual](https://man7.org/linux/man-pages/man2/perf_event_open.2.html).

## Other benchmarks

- `222CpuWeightSchedulingBenchmark.py`: paired comparisons of weight scheduling,
  response-cache traversal and shared input evaluation on a configured model.
- `210CpuBackendBenchmark.py` / `211MpsBackendBenchmark.py`: synthetic CPU/MPS
  comparisons.
- `221BinningBackendBenchmark.py`: histogram binning and smoothing comparisons.
