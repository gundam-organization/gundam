# Performance benchmarks

These scripts measure performance; they are not part of the regular regression
suite. Use a Release build with the Python interface enabled and select the
matching installed `GUNDAM` Python module.

## Resident MPS profiling with production inputs

Build Release with `WITH_PYTHON_INTERFACE=ON`, then select that build's Python
module and matching interpreter. Example from the repository root:

```sh
env -u DYLD_LIBRARY_PATH \
  PYTHONPATH="$PWD/cmake-build-release/src/PythonInterface" \
  python3.15 tests/benchmarks/225MpsBackendProfiling.py \
  --work-dir /Users/nadrino/Documents/Work/Output/results/gundam/GundamInputOA2024 \
  --config configOA2024.yaml --threads 4 \
  --override override/v13ProdFewRuns.yaml override/noSimplex.yaml override/backendMpsTest.yaml
```

On macOS an existing `DYLD_LIBRARY_PATH` containing an installed GUNDAM can
redirect loading even when Python reports a module path inside the build tree.
Removing it for this invocation selects the local build with these diagnostics;
other installations may need their own dependency search paths.

This reproduces the Asimov input setup (`-a`) and overrides, then measures
repeated likelihood evaluations without running a fit or HESSE. Plotting, PCA,
and parameter-step scans are disabled. Input selections and event limits are
preserved. MPS is explicitly selected while its backend options are retained.
The data is loaded once, and event weights/histograms stay resident during
measurements as in the fit loop. Four threads control host work/loading, not
the number of GPU threads.

Defaults run constant, one-coordinate, two-coordinate, and all-coordinate
workloads, with two timed trials each. Local trials cover the complete effective
fit-coordinate list (including eigen coordinates); their call count increases
past `--calls 30` if needed. Global trials use at most `--global-calls 10`.
`--probe-calls 8` adds separate reference/probe passes per workload; these short
passes sample coordinates across the fit list. All proposals are seeded, bounded
by the parameter domain, and use a standard deviation of 0.1 prior sigma.

Output under `mps-profiling-<timestamp>-<PID>/` contains:

- `settings.json`, `resolved-fitter-config.json`, `parameters.json`: invocation,
  resolved configuration, and effective coordinate manifest.
- `model.json`: selected GPU/path, buffer lengths and Shared/Private modes,
  SIMD width, dispatch sizes, dial types and events grouped by dial count.
- `summary.json`: ordinary evaluation, CPU preparation, GPU command, host wait,
  upload, readback and likelihood statistics; separate GPU stage probe timings.
- `*-trial*.json`: raw ordinary measurements, likelihoods, transfer volumes,
  parameter-sequence digests and final histograms.
- `*-probes.json`: separate synchronized GPU stage measurements, validated
  against replaying the same sequence without profiling (2e-5 float tolerance).
- `native.log`: complete native output, with supervisor heartbeats every 30 s.

The packed-weight path probes cached responses, event weights and histogram
reduction plus readback. The event-propagation path probes individual dispatched
kernels, including binning/index construction when parameters trigger them.
Stages not dispatched are absent. Probe passes split stages into commands and
wait after each one: their durations include different scheduling/cache behavior
and must not replace ordinary timings. The histogram/readback group on the
packed path includes GPU buffer copies. `devicePropagationSeconds` in an ordinary
trial measures the full command buffer, including GPU copies. CPU readback time
measures subsequent host conversion, not those copies. CPU wait and GPU duration
overlap; do not add all timers. Zero stage timers in ordinary packed-weight
trials mean those stages were not separately timed.

These probes report elapsed GPU time and layout, not hardware cache misses,
achieved memory bandwidth, or occupancy. Summaries are checkpointed after each
ordinary trial. `--probe-calls 0` disables the separate stage pass.

### Comparing compact cached dial dispatches

Add `--compare-cached-dials --comparison-rounds 2` to the MPS command above to
compare the original full descriptor scan with dispatching only cached dial
indices. Both modes reuse one loaded model and retain the same response slots,
event order, and arithmetic. The compact mode is the production default; the
full scan remains available through a diagnostic Python switch for A/B runs.

Each round runs identical seeded sequences for both modes, alternating A/B and
B/A order between trials/rounds. Every measured likelihood and each final
histogram must match **exactly**. With the default scenarios and 842 effective
coordinates, two rounds of two trials produce 10,424 ordinary evaluations.
Synchronized GPU stage probes run after all ordinary A/B measurements and are
also compared exactly. Input selection and cached-dial reuse thresholds stay
fixed throughout the comparison.

`model-fullScan.json` and `model-compactIndices.json` report selected modes,
dispatched dial counts, and the extra cached-index buffer storage. Files named
`round*-*-trial*-*.json` hold all raw ordinary passes. `summary.json` reports
median/mean/p95 times by scenario/mode, speedup ratios, completion status, and
exact comparison status. The diagnostic switch affects the packed-weight MPS
path; the event-propagation path uses a different dial executor.

### Comparing compact dial descriptors

Use `--compare-descriptors --comparison-rounds 2` in place of
`--compare-cached-dials` to compare the legacy 48-byte spline / 36-byte
normalization and external-weight descriptors against compact 24-byte / 16-byte
records with a separate smoothing table. Both modes retain compact cached dial
dispatches from the previous optimization. Model loading enables
`backendConfig.enableDiagnosticDescriptorComparison` so both layouts are
available; normal runs allocate only the compact layout.

Descriptor comparisons use the original payload order in both modes to isolate
the descriptor-size experiment. The same alternating order, parameter sequences,
ordinary timing passes, separate stage probes, and exact comparisons apply. Modes are called
`legacyDescriptors` and `compactDescriptors` in the JSON files. A/B allocation
contains both layouts: subtract the inactive descriptor buffers when comparing
normal production memory requirements. The smoothing table belongs only to the
compact representation. No event, coefficient or reference reordering is
performed by this experiment.

### Comparing spline payload order

Use `--compare-payloads --comparison-rounds 2` to compare the original payload
order with first-use order across logical 32-event GPU tiles. Both modes use
compact descriptors and compact cached-dial lists. Payload values, descriptor
IDs, response slots, event references, and multiplication order are preserved;
only payload offsets and placement differ.

The script sets `backendConfig.enableDiagnosticPayloadComparison` before model
initialization. The A/B model retains both payload buffers and both sets of
compact descriptors so mode changes do not reload the dataset. Production uses
the reordered payloads by default and allocates only one layout.

JSON mode names are `originalPayloads` and `orderedPayloads`. Ordinary timing
trials alternate A/B order; every likelihood and final histogram must match
exactly. Separate GPU stage probes follow all ordinary trials. Do not interpret
the dual A/B allocation as production GPU memory consumption.

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

## Response locality diagnostic

After rebuilding/installing the C++ library and Python interface, run:

```sh
python3 "$REPO_DIR/gundam/tests/benchmarks/223CpuWeightScalingBenchmark.py" \
  --config configOA2024.yaml --threads 8 --workers 8 --locality-only \
  --override override/v13Prod.yaml override/backendCpuTest.yaml
```

This loads the full configured dataset once with eight loading threads, primes
the model, then writes `locality.json`. There is no fit, sampling pass or repeated
timing trial. Alternatively add `--locality-probe` to a normal scaling or
single-thread benchmark; analysis runs after measurements to avoid warming them.

The diagnostic walks actual event descriptors and dial references in serial
event-weight order. It compares current response slots against a hypothetical
numbering by first use, retaining the current buffer's page offset for both.
No response values, dial payloads, references or multiplication order are changed.
Mixed cached/uncached models report unsupported rather than assuming all accesses
are response-cache loads.

Full-stream metrics include address-jump histograms, consecutive accesses sharing
a 64-byte line or 4096-byte page, line runs, total distinct lines and distinct
lines per 4096-reference window. Reuse distances count distinct intervening
lines exactly on the **first one million references**, using a bounded temporary
Fenwick tree. The resulting `idealLru` entries estimate cold, fully associative
response-only LRU hits for 32 KiB, 256 KiB, 2 MiB and 32 MiB capacities. They do
not model actual cache hierarchy, associativity, prefetching, other buffers or
concurrent workers. The prefix may be unrepresentative of later samples.

This is a geometry diagnostic, not a speed prediction or a hardware miss count.
The simulated numbering is never executed. Temporary storage consists of one
32-bit mapping per response slot, one last-position entry per possible cache line
and at most four MB for prefix reuse analysis; the report estimates working bytes.
These buffers are released on return and no full-size response copy is created.

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

## Production event rebinning (CPU and MPS)

`226ProductionBinningProfiling.py` loads the same production configuration in a
separate process for each backend and smoothing mode. Example:

```sh
env -u DYLD_LIBRARY_PATH \
  PYTHONPATH="$PWD/cmake-build-release/src/PythonInterface" \
  python3.15 tests/benchmarks/226ProductionBinningProfiling.py \
  --work-dir /Users/nadrino/Documents/Work/Output/results/gundam/upgrade-cc-2026 \
  --config ndUpCcInc.yaml --threads 4 \
  --override override/datasetHighlandv5.27.1_CCLyon.yaml override/ebMigration.yaml \
    override/binResponse.yaml override/backendMpsTest.yaml
```

The default parameter set is `Cross-Section Systematics Eb`; use
`--parameter-set` to select another set. Constant, one-coordinate and
all-coordinate sequences vary only that set, by 0.1 prior sigma with bounded,
seeded proposals. Other parameters remain at their initialized values. These
are timing workloads, not an actual MCMC chain. `--calls` controls evaluations
per trial; choose a multiple of the parameter count to cover coordinates evenly.
`--probe-calls` controls separate correctness/profiling passes; choose at least
the parameter count to check every local coordinate. Defaults are 10 calls,
two trials and five probe calls.

The `smoothed` case preserves the configured binning response; `hard` removes
sample `binningResponse` entries from the expanded configuration. Both use
Asimov data initialized in their respective mode. Input files, event selections
and external workers are preserved. No fit, HESSE, PCA, plotting or step scan is
run. `--backends` and `--modes` select subsets of the four cases.

Each case writes resolved configuration, effective parameter manifest, raw
timings/counters, summary, separate reference/probe passes and a native log.
MPS cases additionally write the GPU model and individual kernel durations.
CPU timings distinguish variable updates, bin searches and histogram filling.
`hostPreparationSeconds` includes dial-state preparation and any Python external
worker evaluation; compare it to full evaluation time before attributing a
bottleneck to the GPU. CPU wait overlaps GPU execution, and synchronized stage
probes change scheduling: neither should be added to ordinary GPU timings.

Correctness passes compare CPU/MPS likelihoods and every reference histogram
(including smoothing mass outside the bins), with relative/absolute tolerances
of 2e-5. Stage profiling is replayed against an ordinary MPS reference. All
histogram materialization happens outside evaluation timers. Final hard MPS
passes materialize event state separately to count actual hard-bin changes;
these passes are excluded from performance statistics. Their
`sameBinFractionAmongRebinned` estimates the opportunity to reuse the previous
bin, rather than measuring an implemented optimization. Event readbacks are
forbidden during the ordinary measurement passes.

### Comparing previous-bin reuse without an external worker

For the upgrade inputs, `override/pmuScaleBinningTest.yaml` defines the
`Pmu scale binning test` parameter set, with a unit normalization on `Pmu` and
`stdDev: 0.01`. It disables the Eb parameter set and replaces the
`ebMigration.yaml` override. Include `binResponse.yaml` to benchmark the
configured smoothing too; omit it for an ordinary fitter run with strict bins.
The production benchmark still needs it when requesting both modes:

```sh
env -u DYLD_LIBRARY_PATH \
  PYTHONPATH="$PWD/cmake-build-release/src/PythonInterface" \
  python3.15 tests/benchmarks/226ProductionBinningProfiling.py \
  --work-dir /Users/nadrino/Documents/Work/Output/results/gundam/upgrade-cc-2026 \
  --config ndUpCcInc.yaml --threads 4 \
  --override override/datasetHighlandv5.27.1_CCLyon.yaml \
    override/pmuScaleBinningTest.yaml override/binResponse.yaml override/backendMpsTest.yaml \
  --parameter-set 'Pmu scale binning test' --step-scale 1 \
  --scenarios constant global --calls 150 --trials 4 --warmup 20 \
  --probe-calls 12 --compare-previous-bin
```

`--step-scale 1` proposes at one prior sigma (1% here). The override resides in
the external input directory. `--compare-previous-bin` adds alternating A/B and
B/A trials on each resident MPS model. The reference disables the shortcut; the
other mode restores the enabled production default. Each trial replays the same
seeded proposals from the initialized point. Results are written to
`previous-bin-comparison.json`, `*-ab*-treeSearch.json`,
`*-ab*-previousBin.json` and separate `*-ab-*-probes.json` files. A/B checks cover
every likelihood and each final histogram with the existing 2e-5 tolerances.
Counter differences are expected: a failed previous-bin check counts as an
additional candidate before the fallback tree search. Smoothed cases keep the
same binning algorithm in both diagnostic modes.

### Large Pmu variations and interval rejection

The external input override `override/pmuScaleBinningStressTest.yaml` is applied
**after** `pmuScaleBinningTest.yaml`. It sets `stdDev: 0.5` and expands
`parameterLimits` to `[0.01, 3.0]`. Keep `--step-scale 1` to propose at 50% of the
nominal unit scale. This is a stress workload; proposals are redrawn outside
the positive domain, so the accepted distribution is a truncated Gaussian.
Simply setting `--step-scale 50` with the original `[0.8, 1.2]` bounds would not
exercise large migrations.

Use the command above with the additional stress override and
`--compare-binning-intervals` in place of `--compare-previous-bin`. For example,
`--scenarios global --calls 200 --trials 6 --warmup 30 --probe-calls 16` gives
1,200 ordinary evaluations per backend/mode plus six alternating A/B pairs on
each MPS model. The previous-bin optimization stays enabled in both variants.

This comparison writes `interval-pruning-comparison.json`,
`*-ab*-fullIntervals.json`, `*-ab*-prunedIntervals.json` and corresponding
separate stage probes. `fullIntervals` executes the original complete interval
loop; `prunedIntervals` stops once a candidate cannot match or contribute. Each
raw row now includes `parameterValues` in parameter-manifest order, after the
evaluation timer, so the actual proposal range can be inspected. Candidate
counts describe bins visited, not individual interval operations; early
interval rejection does not reduce these counts.


### Comparing resident propagation optimizations

Use `--compare-resident-stages` instead of the other comparison flags to rotate
five variants on each resident MPS model:

| Variant | Dial/weight cache | Contribution-index reuse | Parallel histogram chunks | SIMD counters |
|---|---|---|---|---|
| baseline | off | off | off | off |
| cached | on | off | off | off |
| indexed | on | on | off | off |
| parallel | on | on | on | off |
| aggregated | on | on | on | on |

The baseline disables the new features in the current executable; it shares the
refactored shader code, rather than loading an older binary. These controls do
not disable previous-bin reuse or interval pruning. All five variants replay the
same seeded coordinates, and the original default is restored afterwards.
Results are in `resident-stages-comparison.json`, per-trial variant files and
separate probes. Probe rows also include `executionState` (evaluated dials,
weight-update flag, index-rebuild flag). This cheap state query and all histogram
snapshots happen after the evaluation timer.

Use the 1% or 50% command above with this flag. To isolate fraction changes while
keeping support lists stable, use the original 1% override with
`--step-scale 0.00001`. This deliberately tiny workload checks index reuse; it
does not represent an ordinary full-sigma proposal. To exercise simultaneous
weight and observable changes, `--parameter-set '*' --step-scale 0.1` selects
all eligible effective fit coordinates. Quote the asterisk to prevent shell
expansion. This mixed workload is useful for checking that dense updates retain
the full dial scan efficiently. Model manifests record the actual coordinate
count. CPU comparisons are performed only when CPU is included in `--backends`.

## Original/CPU/MPS numerical comparison

`227BackendComparison.py` replays the same parameter proposals on separate
original, CPU and MPS models, validating every likelihood, histogram bin content
and bin error. MPS runs on macOS; other platforms compare original and CPU.
It also records evaluation timings, excluding explicit histogram materialization.
For example:

```sh
env -u DYLD_LIBRARY_PATH \
  PYTHONPATH="$PWD/cmake-build-release/src/PythonInterface" \
  python3.15 tests/benchmarks/227BackendComparison.py \
  --work-dir /path/to/inputs --config config.yaml --threads 4 \
  --override override/dataset.yaml override/enableEb2d.yaml \
  --output /tmp/gundam-backend-comparison
```

See [multi-input dials](../../docs/backends/MultiInputDials.md) for the surface
implementation and test coverage. This benchmark does not run a fit or Hesse.
