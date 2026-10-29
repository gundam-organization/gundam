# Performance benchmarks

These scripts measure performance; they are not part of the regular regression
suite. Use a Release build with the Python interface enabled and select the
matching installed `GUNDAM` Python module.

`--output` is optional. By default, scripts create a directory in the launch
directory named `benchmark-<script-name>-YYYYMMDDHHmmss`, using local time and
the script name without its extension. An explicit `--output` path takes
precedence. The single-thread wrapper uses its own script name; the CUDA shell
wrapper uses the name of the Python profiler it launches. Existing directories
are not overwritten. For compatibility, `221BinningBackendBenchmark.py` also
accepts an explicit `.json` output file; directory outputs contain `summary.json`.

## CPU formulas and composites with Eb migration

`230CpuFormulaCompositeBenchmark.py` replays the five physical Eb parameters on
the Upgrade configuration. Use the same data and migration overrides as the fit,
then append `onlyEb.yaml` to isolate those dials:

```sh
env -u DYLD_LIBRARY_PATH \
  PYTHONPATH="$PWD/cmake-build-release/src/PythonInterface" \
  python3.15 tests/benchmarks/230CpuFormulaCompositeBenchmark.py \
  --work-dir /path/to/upgrade-cc-2026 --threads 4 \
  --output /path/to/new-eb-benchmark \
  --override override/datasetHighlandv5.27.1_CCLyon.yaml \
    override/ebMigrationDial.yaml override/binResponse.yaml \
    override/backendCpuTest.yaml override/onlyEb.yaml
```

The model is loaded once as Asimov data. PCA initialization, parameter-step
response scans, plotting and sampled OS profiling are disabled. No minimization
or Hesse is run and no event limit is added. Constant, individual-coordinate and
all-Eb changes alternate between ±0.1 and ±0.2 prior sigma. Defaults use three
trials of 160 calls per scenario with eight warmups, rotating scenario order.

The output contains the resolved configuration, actual loaded module and hash,
parameter manifest, per-call timings/counters/likelihoods, and median/p95
summaries. Histogram digests are captured after each trial; event weight, Pmu
and bin digests are captured after the final trial of each scenario. All host
materialization and digest calculation occur outside the timed evaluations.
Separate baseline and candidate processes can therefore compare exact outputs
as well as timing. The script assumes the five named physical Eb coordinates
and loaded `Pmu` variable of this configuration.

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
`--propagation-mode cachedResponses` or `--propagation-mode recomputeResponses`
overrides the configured response strategy. The backend default is
`cachedResponses`; packed-layout comparisons explicitly select
`recomputeResponses`, because their diagnostic switches belong to that executor.
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

The MPS benchmarks measure the production implementation: compact cached-dial
indices and descriptors, ordered spline payloads and resident propagation
optimizations. Temporary switches comparing these algorithms with their older
implementations have been removed. Ordinary timing passes and separate GPU
stage probes remain available.

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

Output under `benchmark-224CpuSingleThreadBenchmark-YYYYMMDDHHmmss/` includes the normal JSON
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

Results are written under `benchmark-223CpuWeightScalingBenchmark-YYYYMMDDHHmmss/`:

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

### Production binning with Pmu variations

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
  --probe-calls 12
```

`--step-scale 1` proposes at one prior sigma (1% here). The override resides in
the external input directory. Previous-bin reuse is always active for certified
strict bins; smoothed cases use the full binning algorithm.

### Large Pmu variations and interval rejection

The external input override `override/pmuScaleBinningStressTest.yaml` is applied
**after** `pmuScaleBinningTest.yaml`. It sets `stdDev: 0.5` and expands
`parameterLimits` to `[0.01, 3.0]`. Keep `--step-scale 1` to propose at 50% of the
nominal unit scale. This is a stress workload; proposals are redrawn outside
the positive domain, so the accepted distribution is a truncated Gaussian.
Simply setting `--step-scale 50` with the original `[0.8, 1.2]` bounds would not
exercise large migrations.

Use the command above with the additional stress override. For example,
`--scenarios global --calls 200 --trials 6 --warmup 30 --probe-calls 16` gives
1,200 ordinary evaluations per backend/mode, followed by separate GPU stage
probes. Interval pruning and contribution-index reuse are always active.
Candidate counts describe bins visited, rather than individual intervals.

To exercise simultaneous weight and observable changes,
`--parameter-set '*' --step-scale 0.1` selects all eligible effective fit
coordinates. Quote the asterisk to prevent shell expansion. This mixed workload
also exercises the full dial scan for dense updates. CPU comparisons are
performed when CPU is included in `--backends`.

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

## CPU model memory

`228CpuMemoryBenchmark.py` loads the full selected Asimov model, records current
and peak RSS (plus malloc statistics on macOS), and exports CPU buffer sizes and
capacities. It replays seeded constant, local and global parameter changes,
without fitting or Hesse. Parameter setters, memory probes and histogram copies
are outside the evaluation timers. PCA, step-response initialization and plots
are disabled; input files and the event selection are preserved.

Run each version in a fresh process, selecting the corresponding compiled Python
module through `PYTHONPATH`. On macOS, sanitize `DYLD_LIBRARY_PATH` so that an
installed module cannot replace the selected build. The script verifies the
loaded module path and records its hash.

```sh
env -u DYLD_LIBRARY_PATH \
  PYTHONPATH=/path/to/build/src/PythonInterface \
  /path/to/venv/bin/python tests/benchmarks/228CpuMemoryBenchmark.py \
  --work-dir /path/to/inputs --config configOA2024.yaml --threads 4 \
  --override override/v13ProdFewRuns.yaml override/backendCpuTest.yaml \
  --output /tmp/oa2024-memory-after --reference /tmp/oa2024-memory-before
```

Omit `--reference` for the first run. With a reference, the script checks the
resolved configuration, parameter manifest, parameter-sequence digests, every
likelihood and final histogram contents/errors exactly, and writes
`comparison.json`. RSS depends on page residency and macOS compression; buffer
capacity and allocator reservations must not be interpreted as resident bytes.
## CUDA production profiling

`229CudaBackendProfiling.py` profiles the full selected production model using
Asimov data, without Migrad, Hesse, PCA, plots or step-response scans. By default
(`--execution-mode resident`) it loads the model once in one worker process,
then replaces only the backend between cases, reusing the immutable layout,
events, dials and Asimov data. This replacement requires fixed bins and no
observable responses; use `--execution-mode isolated` for migration or smoothing.
The previous backend's buffers are released before
the next backend is built. Parameters are restored to their initial effective
fit coordinates; each new backend is primed and warmed up outside measurements.
Default cases are CUDA `cachedResponses` and `recomputeResponses`, both with
interleaved indices (`-interleaved` case names); add `CPU`
for a numerical and performance reference on the same machine and thread count.
Backend-specific configuration is replaced for each case; input files, selections
and dataset limits from the config/overrides are preserved.

Build with `WITH_PYTHON_INTERFACE=ON`, `WITH_BACKENDS=ON` and `WITH_CUDA_LIB=ON`.
Use the Python interpreter matching the compiled GUNDAM module and an allocated
GPU at the CC. From the repository root, for example:

```sh
PYTHONPATH=/sps/t2k/ablanche/build/gundam/src/PythonInterface${PYTHONPATH:+:$PYTHONPATH} \
python3 tests/benchmarks/229CudaBackendProfiling.py \
  --work-dir /path/to/GundamInputOA2024 \
  --config configOA2024.yaml \
  --override override/v13Prod.yaml override/backendCudaTest.yaml \
  --cases CUDA-cachedResponses CUDA-recomputeResponses CPU \
  --threads 4 --calls 30 --global-calls 10 --trials 2 --warmup 5 \
  --output /sps/t2k/ablanche/work/results/cuda-oa2024-profile
```

### CUDA migration and smoothing on upgrade-cc-2026

Use the same script with `--execution-mode isolated`: dynamic/observable models
cannot currently use resident backend replacement. Each case loads the model
once; loading and warmup are outside evaluation timings and `setup.json` records
initialization separately. This benchmark uses Asimov data (the fitter's `-a`)
without running a fit or Hesse, with all input selections/limits preserved.

For the supplied upgrade configuration, `pmuScaleBinningTest.yaml` defines
`Pmu -> pmuScale * Pmu`, with prior 1 and sigma 0.01. Applied after
`enableEb2d.yaml`, it disables `Cross-Section Systematics Eb`. `binResponse.yaml`
smooths Pmu with fixed width **1 MeV**, not a fractional width. Therefore the
following workload targets Pmu scaling and smoothing; it does not activate Eb
migration. Other fit parameters stay at their initial values.

From the repository root in a GPU allocation, adapt `--work-dir` to the CC input
location if needed. Use the Python matching the compiled module:

```sh
PYTHONPATH=/sps/t2k/ablanche/build/gundam/src/PythonInterface${PYTHONPATH:+:$PYTHONPATH} \
python3 tests/benchmarks/229CudaBackendProfiling.py \
  --work-dir /sps/t2k/ablanche/work/results/gundam/upgrade-cc-2026 \
  --config ndUpCcInc.yaml \
  --override override/datasetHighlandv5.27.1_CCLyon.yaml \
    override/disable2p2hNormNuBar.yaml override/enableEb2d.yaml \
    override/pmuScaleBinningTest.yaml override/binResponse.yaml \
    override/backendCudaTest.yaml \
  --cases CUDA-cachedResponses-interleaved CUDA-recomputeResponses-interleaved CPU \
  --threads 4 --execution-mode isolated \
  --parameter-match pmuScale --scenarios constant local1 \
  --step-scale 1 --calls 30 --trials 3 --warmup 5 --require-rebinning \
  --output ./cuda-upgrade-binning-profile
```

`--parameter-match` is a regex over eligible effective parameter full titles;
no match is an error, and the selected coordinates are recorded in `parameters.json`.
`--step-scale 1` proposes a seeded Gaussian displacement of one prior sigma:
`pmuScale = 1 + N(0, 0.01)`, respecting its domain, without a random walk.
`constant` repeats identical parameters; `local1` changes only the selected
coordinate. `--require-rebinning` rejects a nonconstant scenario if no measured
call ran event binning. This checks activity, not a count of actual hard-bin
crossings. With only one selected coordinate, do not request `local2`.

Start with `report.md`: it includes observables/search, bin index, histogram,
readback and total evaluation times, plus rebin/contribution counters and CUDA
contribution-buffer capacity. CUDA's variable stage includes the fused hard-bin
search, while its bin-index stage includes fractional evaluation, prefix scans,
allocation synchronization and stable sorting. CPU stage boundaries differ;
compare total evaluations first. GPU times overlap host evaluation, so they
must not be added together. Event state/weight readbacks are rejected during
measurement; final histogram/outside materialization is outside the timers.

`comparison.json` checks every likelihood and each trial's final normal and
outside histogram sums/errors against CPU. Input proposals/digests are identical
across cases. This is propagation performance, not full-fit throughput, and the
comparison does not validate covariance matrices. The output directory must be new.

The [CC production results from 2026-10-25](../../docs/backends/CudaUpgradeBinningBenchmark.md)
pass both CUDA comparisons against CPU. Varying pmuScale takes 4.10 ms with
cached CUDA, 5.09 ms with recompute CUDA and 29.93 ms on four CPU threads.

For large Pmu displacements, insert `override/pmuScaleBinningStressTest.yaml`
immediately after `override/pmuScaleBinningTest.yaml`. This changes the proposal
sigma to 0.5 and domain to `[0.01, 3]`, preserving the 1 MeV smoothing width.
Use a fresh output directory. To evaluate the ordered CUDA bin hierarchy,
repeat the normal and stress workloads with outputs
`cuda-upgrade-binning-indexed-profile` and
`cuda-upgrade-binning-stress-indexed-profile`; both previous runs are documented
in the [production results](../../docs/backends/CudaUpgradeBinningBenchmark.md).

For a first quick pass, use `--calls 10 --trials 1 --warmup 2`; each selected case
still pays the complete dataset loading cost. To locate expensive coordinates
across the full model afterward, omit `--parameter-match` and use
`--scenarios constant local1 global` (local1 covers all eligible coordinates).

### General profiling settings

The output directory must be new. The same seeded proposals are replayed in
each case. Scenarios are `constant`, `local1`, `local2`, and `global`. Local
scenarios always cover all eligible fit coordinates, so `--calls 30` is a
minimum: 842 coordinates require at least 842 local1 and 421 local2 calls per
trial. The shared CPU/MPS proposal helper selects enabled, unfixed coordinates
with enabled penalties, using effective eigen coordinates when configured.
The default proposal scale is 0.1 prior sigma. This workload helps locate
propagation costs; it does not reproduce the exact call distribution of Migrad.

For a shorter first pass, use `--scenarios constant global --calls 10
--global-calls 10 --trials 1 --warmup 2`; initialization still loads the full
selected model. CPU is optional and requires building its own buffers, but does
not reread the dataset in resident mode. Use `--execution-mode isolated` for
one fresh process and dataset load per case, as in the initial OA2024 benchmark.
The mode is recorded in settings and reports; allocator and OS/device cache
state may differ between modes. The order is the supplied `--cases` order.
Use a second run with `--threads 8` if you want the eight-thread CPU reference;
thread count applies to every selected case in a run.

Read `report.md` first, then `summary.json` and each case's
`local1-coordinate-costs.json`. Raw per-call measurements, resolved config,
parameter manifest are saved in each case directory. Resident mode writes one
shared `native.log` at the output root, plus per-case `setup.json` recording
setup time, PID and whether dataset loading was included. Isolated mode keeps
one native log per case. Hardware,
Slurm environment, source commit and loaded module path/hash are recorded.
An unavailable backend/device fails the benchmark, rather than producing a
successful report without GPU measurements.

CUDA uses existing stream events to time parameter upload/error reset, dial
responses, event weights and histogram reduction. No extra stage synchronization
is introduced by the script. It also measures full evaluation, parameter setters,
host preparation, histogram readback and host statistical likelihood. Explicit
histogram materialization is outside evaluation timers, and event readbacks are
forbidden during measurements. Histogram readback needed by the host likelihood
remains part of evaluation time.

GPU total overlaps the synchronous host evaluation; do not add those timers.
The host preparation includes propagation input preparation; the remaining
evaluation time may include penalty evaluation, launches, synchronization and
other fitter bookkeeping. CUDA `deviceWaitSeconds`/`commandEncodeSeconds` are
not currently populated, so this script does not present them as measurements.
The response timer covers both response modes despite its API name
`cachedDialStageSeconds`. GPU stage rankings point to candidates for further
profiling; they do not measure bandwidth, cache misses or occupancy.

Cached CUDA now indexes dials by their original scalar parameter. After cache
priming, constant inputs skip the response kernel; sparse changes visit only
the affected parameter ranges. Updates affecting more than half of all dials
use the dense traversal. `cachePreparationSeconds` measures host selection time,
and `medianResponseCandidates` in case summaries and coordinate costs reports
the number of dials visited, before transformed-input reuse. Per-call candidate
counts are also saved. These counters describe work selection, not the number
of responses whose numerical value changed. The index costs four device bytes
per nonconstant dial, plus small host/device parameter-range buffers.

For the follow-up OA2024 run, keep the original `cuda-oa2024-profile` directory
and use `--output ./cuda-oa2024-profile-sparse`. Rebuild and reinstall first if
your Python module is loaded from the installed prefix: the previous CC run
used `/sps/t2k/ablanche/install/gundam/lib`, rather than the build tree. Keep
the same inputs, proposal settings, four-thread allocation and GPU model.
Compare numerical agreement first, then local/constant response timings and
candidate counts. The completed isolated sparse CUDA passes returned from the
CC match the prior CPU reference; see the [OA2024 results](../../docs/backends/CudaOa2024Benchmark.md#sparse-production-results).
The sparse snapshot lacks its new CPU result. The later complete
[`cuda-oa2024-event-interleave` run](../../docs/backends/CudaOa2024Benchmark.md#resident-production-run-with-interleaved-indices)
validates all five cases, including CPU, and resident replacement on the CC V100.

`comparison.json` checks every measured likelihood, proposal sequence and final
histogram contents/errors of every trial (relative `1e-9`, absolute `1e-7`). It
reports speedups against CPU when included, otherwise against the first case.
It fails on a numerical mismatch and leaves raw results available for diagnosis.
Timing differences are never a pass/fail criterion. The isolated CUDA passes
have been exercised at the CC; the CPU path was exercised locally
on a small two-parameter model. Resident CPU replacement was also checked
through the Python driver and native tests, including repeated replacements
after deleting the input file, stale deferred results, build failure and recovery.
Optional CUDA regular tests cover CPU/CUDA/CPU/CUDA replacement in both modes.

### Event-weight index layout experiment

CUDA `backendConfig.weightLayout` accepts `eventMajor` and
`warpInterleaved` (default). The latter stores the first dial index of 32 events
together, followed by their second index, and so on. Adjacent GPU threads can
load adjacent index locations. Each event still multiplies responses in its
original order; base weights, reweight caps, result slots and bin membership
are preserved. This targets index-load locality without changing reductions.
The response values themselves may still require scattered accesses.

Padding is capped at twice the original event-dial index count. Models exceeding
that cap retain `eventMajor`. `model.json` reports requested and actual layouts,
original/packed counts and packed index bytes. The cap applies to this one
buffer, not total device memory. The completed OA2024/V100 A/B run measures
about 22% less cached local evaluation time (3.080 to 2.399 ms) and 66.96 MiB
additional index storage, with identical recorded CUDA likelihoods/histograms
between layouts. CUDA defaults to `cachedResponses` with `warpInterleaved`.
Explicit `eventMajor` remains available for comparisons, and the memory cap
still falls back automatically to that layout for highly ragged models.
The benchmark preserves historical case meanings: names without `-interleaved`
explicitly select `eventMajor`, while names with that suffix select
`warpInterleaved`. With no `--cases`, both interleaved response modes are selected.

The extra cases select this option without changing response propagation mode:

```sh
python3 tests/benchmarks/229CudaBackendProfiling.py \
  --work-dir /sps/t2k/ablanche/work/results/gundam/GundamInputOA2024 \
  --config configOA2024.yaml \
  --override override/v13Prod.yaml override/backendCudaTest.yaml \
  --cases CUDA-cachedResponses CUDA-cachedResponses-interleaved \
    CUDA-recomputeResponses CUDA-recomputeResponses-interleaved CPU \
  --threads 4 --execution-mode resident \
  --output ./cuda-oa2024-profile-weight-layout
```

All five cases share the loaded model. Compare event-weight stage and complete
evaluation durations alongside actual layout/index bytes in `report.md` and
numerical checks in `comparison.json`. A faster response stage is not expected
from this change. GPU tests check equality with the original layout for ordered
products, histogram sums/errors and likelihood, and optional regular tests
compare both propagation modes against the CPU fit/Hesse covariance reference.

## OA2024: graph search loop versus unrolled (issue #1039)

`239CudaGraphSearchBenchmark.py` wraps the existing production profiling driver.
It uses `configOA2024.yaml`, `override/v13Prod.yaml` and
`override/backendCudaTest.yaml` from a copy of `GundamInputOA2024` (the local
source inputs are `/Users/nadrino/Documents/Work/Output/results/gundam/GundamInputOA2024`).
Pass the corresponding **CC directory**, with its existing data paths/environment.
No input files are modified or converted from splines to graphs.

The temporary CMake option `GUNDAM_GRAPH_UNROLLED_SEARCH` defaults to `OFF`
(the original dynamic loop). `ON` uses explicit offsets `16/8/4/2/1` for up to
32 knots and retains a loop for higher offsets in larger graphs. It changes
only graph interval search, not general-spline search, interpolation, caching
or weight layout. It applies to the shared function used by CacheManager,
CPU/CUDA backends and the embedded MPS kernels. This benchmark measures the
**CUDA backend**, not the legacy CacheManager execution path.

Run both passes in the same GPU allocation, using the same environment, Python,
checkout, inputs, CPU affinity and GPU. The commands below reuse an existing
Release build; preserve its toolkit, architecture and dependency settings.
Use the Python interpreter matching the build. `--build-dir` selects its
Python module and checks loaded GUNDAM library paths on Linux, rejecting an
installed library outside that build. Sanitize `LD_LIBRARY_PATH` if necessary.
Compilation must finish before each run; the recorded CMake cache alone cannot
prove that a stale binary was rebuilt.
The CUDA diagnostics also report the compiled search option and graph dial
count. The wrapper checks them after measurement and exits nonzero if the
binary disagrees with the cache or the workload contains no graph dials.

```bash
repo=$PWD
buildDir=/sps/t2k/ablanche/build/gundam
inputDir=/sps/t2k/ablanche/work/results/gundam/GundamInputOA2024
resultDir=/sps/t2k/ablanche/work/results/graph-search-ab

# First pass: original loop.
cmake -S "$repo" -B "$buildDir" -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA_LIB=ON -DWITH_BACKENDS=ON -DWITH_PYTHON_INTERFACE=ON \
  -DGUNDAM_GRAPH_UNROLLED_SEARCH=OFF
cmake --build "$buildDir" -j4
python3 tests/benchmarks/239CudaGraphSearchBenchmark.py \
  --build-dir "$buildDir" --work-dir "$inputDir" --output "$resultDir/loop"

# Second pass: rebuild, then replay exactly the same proposals.
cmake -S "$repo" -B "$buildDir" -DGUNDAM_GRAPH_UNROLLED_SEARCH=ON
cmake --build "$buildDir" -j4
python3 tests/benchmarks/239CudaGraphSearchBenchmark.py \
  --build-dir "$buildDir" --work-dir "$inputDir" --output "$resultDir/unrolled" \
  --reference "$resultDir/loop"
```

Both runs refuse to overwrite an existing output directory. Defaults: four host
threads, three trials, ten warmup calls, 100 constant calls, one full sweep of
single-coordinate changes, and 100 global calls per trial, with seeded proposals.
Each run loads the dataset once, then measures both `cachedResponses` and
`recomputeResponses`, using the same `warpInterleaved` layout. No fit or Hesse
is performed; loading, backend replacement, warmup and final histogram extraction
are excluded from evaluation timers. Expect production dataset initialization
to take much longer than the timed passes. Additional driver options are accepted,
e.g. `--global-calls 300 --trials 5`; use identical options for both passes.

Read `unrolled/graph-search-comparison.md`: ratios above 1 mean the unrolled
variant is faster. The comparison checks settings, resolved configurations,
parameter manifests, proposal digests, every measured likelihood and final trial
histogram contents/errors (default rtol `1e-9`, atol `1e-7`); failures exit nonzero.
Detailed measurements are in each case's trial JSON and `report.md`.
`graph-search-build.json` records the option, CMake cache, loaded library paths
and hashes of GUNDAM shared libraries in the selected build, alongside the
profiling driver's existing commit, module hash, GPU and Slurm metadata.

Focus on `cachedDialStageSeconds` for global changes and for recompute mode:
these force response evaluation. Constant cached calls can skip graph evaluation
entirely and serve as a control. Full-evaluation speedup includes other work;
the response stage itself includes spline and other dial types too. Inspect
`native.log`/model diagnostics for the graph population before interpreting a
null result as evidence about graph search. These timings do not isolate SM
occupancy, memory stalls, or predict whole-fit speedups.

Restore `-DGUNDAM_GRAPH_UNROLLED_SEARCH=OFF` and rebuild after the experiment.
