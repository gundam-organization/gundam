# Automatic CPU event-weight scheduling

The CPU backend automatically builds **two contiguous work blocks per worker**
at initialization. Workers claim blocks dynamically from a shared counter. A
worker that finishes early can process another block rather than waiting for a
worker with a slower event range.

No block-size or block-count tuning is required. For four threads the backend
prepares eight blocks; for eight threads it prepares sixteen. The blocks use the
existing estimate `1 + dialCount` to balance expected work, so event counts per
block may differ. An individual event is never split. Block boundaries remain
fixed after construction, while their assignment to workers can change at each
propagation.

## Defaults and diagnostic comparison

Dynamic scheduling is the default. Single-thread runs and datasets with fewer
than two events per worker retain the fixed path. The actual mode and block
count are printed at construction.

For a diagnostic comparison on the same node and thread count, the previous
fixed worker ranges can still be selected explicitly:

```yaml
fitterEngineConfig:
  likelihoodInterfaceConfig:
    backendManagerConfig:
      isEnabled: true
      useBackend: CPU
      options:
        - name: CPU
          weightScheduling: Fixed
```

Omit `weightScheduling` for normal use. `weightScheduling: Dynamic` explicitly
selects the same automatic default. The former prototype's `weightBlockSize`
option has been removed; it is not a supported configuration field.

## Implementation and numerical behavior

Blocks are prepared once, with boundaries balanced using event/dial counts.
When a very expensive event crosses several estimated-work targets, every later
block still retains at least one event. Small workloads bypass the shared queue.
The extra storage is two range descriptors per worker, aligned worker accounting
and a shared atomic counter. It does not duplicate event, response or weight
buffers or create a queue proportional to dataset size.

Each claim requires one relaxed atomic increment. Inputs are immutable during
the weight job, workers write disjoint result slots and the thread pool waits
for completion before later stages consume weights. The scheduler calls the
existing cached or mixed-response kernel on each block. Per-event multiplication
order, double precision, cap placement and output-slot mapping are preserved.
External weights, constant dials and optional features use the same evaluators.
Histogram jobs and their reduction order are unchanged.

Cache-response updates and histogram filling are not scheduled through this
queue. A single event with many dials remains indivisible. Two blocks per worker
are a coarse balancing policy: they can reduce the observed idle tail but need
not eliminate all imbalance. Even with few claims, locality changes and
synchronization can cause regressions on some workloads. A factor-of-two worker
time difference does not mathematically guarantee that twice as many blocks
will improve every execution.

## Profiling

CPU JSONL metadata records `weightSchedulingRequested`, actual `weightScheduling`,
`dynamicBlockTarget` and actual `weightBlockCount`. `weightBlocks` describes the
actual work ranges with event and dial-reference counts. Sampled worker records
contain `weightEvents` and `weightBlocks` processed by each worker.

The historical `weightRanges` field still describes fixed reference ranges.
`weightRangesRole` is `fixedReference` for dynamic scheduling and `executed` for
the fixed path. Empty fixed reference ranges are omitted. The summary tool
prints the actual mode, block count and per-worker work counts, while retaining
compatibility with older prototype reports that recorded a block size.

Faster workers may process more events while finishing closer to other workers.
Compare the actual per-worker counts and durations rather than assuming a worker
receives the same event range on successive propagations.

## Benchmarking the full input model

`tests/benchmarks/222CpuWeightSchedulingBenchmark.py` compares Fixed and
Dynamic with **2, 5 and 10 blocks per thread** by default. The dataset loads **once per thread count**, and every factor and
round reuses that initialized model. Different thread counts run sequentially in
separate processes. Loading and work-range reconstruction are outside the timed
region. After warmup, the script performs actual likelihood evaluations under four
parameter-change workloads; no minimizer or Hesse is run.

Build with `WITH_PYTHON_INTERFACE=ON` and select the matching GUNDAM Python module.
Run within one cluster allocation, from the input directory, for example:

```sh
python3 /path/to/gundam/tests/benchmarks/222CpuWeightSchedulingBenchmark.py \
  --config configOA2024.yaml --threads 8 --factors 2 5 10 \
  --calls 100 --global-calls 20 --trials 2 --rounds 1 \
  --override override/v12ProdRun45.yaml override/backendCpuTest.yaml
```

Use the overrides selecting the actual full dataset on the cluster. Paths are
resolved relative to `--work-dir` (default: current directory). Keep `--override`
last because it accepts multiple paths. `--threads 4 8` additionally compares
both thread counts within the same allocation; do not exceed allocated CPUs.

With eight threads, the cases are Fixed and Dynamic with 16, 40 and 80 blocks.
The default warmup is five evaluations per case. Non-global trials contain `--calls` evaluations (default 100); global trials
contain `min(--calls, --global-calls)` evaluations, with a default cap of 20. With `--threads 8 --rounds 2`, all 32 cases (four workloads times four layouts times two rounds) share a single
loaded model, with reversed case order in the second round to expose order/load
effects. `--threads 4 8` requires two sequential loads because each backend
constructs its worker pool for one thread count.

Output goes to a timestamp/PID-specific directory unless `--output` supplies a
new directory. Existing directories are rejected to preserve earlier results.
Each completed case writes a JSON with per-trial medians, means, weight p95,
cache/histogram/full-evaluation timings, raw call timings, actual block count,
host, affinity, build revision and model counts. Final histogram sums and errors are
checked against the first case for that workload/thread count with
absolute/relative tolerance `1e-12`. All measured likelihood values are compared
with tolerance `1e-10`, and hashes verify that the parameter sequences match.
`summary.json` is atomically replaced after each successful case and includes
weight speedup relative to Fixed for the same workload, thread count and round. Completed
results therefore survive interruption during later cases. A `.partial.json`
checkpoint is also atomically updated after every completed trial; it is removed
when the final case JSON is written. An interrupted trial may be absent. Collect
the entire output directory, including the native `t<threads>.log` files, when
returning results.

The main log prints initialization, layout changes, warmup, trial starts,
progress approximately every five seconds during trials, and saved results.
Every 30 seconds the parent reports that its child process is still running,
including during silent native loading. This heartbeat confirms process
liveness, not forward progress inside the loader. Full native output is flushed
to the per-thread-count log; only benchmark progress is forwarded to the main log.

### Diagnostic mechanism and measurement scope

The Python diagnostic method
`BackendManager.setDiagnosticCpuWeightScheduling(mode, blocksPerThread)` changes
only the CPU work ranges and selected kernel, between synchronous propagations.
Model buffers, response caches, event weights and histogram storage are retained.
The method returns lightweight JSON layout diagnostics, verifies positive factors
and overflow, and starts a fresh profile run if profiling was already enabled.
A runtime change does not reread the environment variable. This is a benchmark
API, not a new YAML tuning option; normal fits retain the automatic default.

The benchmark disables sampled profiling entirely. Layout diagnostics are read
outside the timed region. It checks for the new API before loading, so an older
Python module fails immediately rather than silently benchmarking unchanged
factors. Dynamic cases with too few events for the requested block count are
rejected rather than timed as Fixed. The earlier environment override remains
available for initial-layout diagnostics but is cleared in benchmark children.

### Parameter-change workloads

All four workloads run by default. Select a subset with, for example,
`--scenarios constant local1 local2` (place it before `--override`).

| Scenario | Change before each timed evaluation |
|---|---|
| `constant` | None: repeatedly evaluate the initialized Asimov point. |
| `local1` | Change one variable fit coordinate, cycling through coordinates. |
| `local2` | Change two variable fit coordinates, cycling through coordinates. |
| `global` | Change every variable fit coordinate at once. |

Local workloads approximate the invalidation pattern of optimizer probes;
`global` approximates an all-coordinate MCMC proposal's propagation workload.
These are neither actual Migrad steps nor an MCMC chain: there is no
accept/reject decision, convergence criterion or posterior sampling claim.
Global points are independent perturbations around the initialized point, not
a random walk. More changed parameters can increase cache-update cost, but
elapsed-time ordering is measured rather than assumed.

The variable coordinates come from enabled parameter sets' effective fit lists;
fixed, disabled and penalty-disabled coordinates are excluded, matching the
fit parameter selection. Eigen-decomposed sets use their
eigen coordinates, consistent with the fit. One eigen-coordinate change may
change many original physical parameters and invalidate many dials. The
`parameters-t<threads>.json` manifest records titles, coordinate spaces, centers
and proposal scales. Existing model conversion rules remain in effect.

Proposals are centered on the initialized coordinate value, with standard
deviation `--step-scale` (default 0.1) times its prior standard deviation; flat
parameters use their step size instead. Values are resampled until finite,
distinct from the previous value and inside the coordinate's configured domain.
A coordinate that cannot be varied produces an explicit error. Local workloads
leave other coordinates at their latest small displacement, so only the selected
one or two coordinates change before each call. Short local runs cover only a
prefix of the coordinates; use enough calls for the full parameter list.

Every trial restores the same center, primes the cache outside timing, and
replays an identical sequence for every layout and round. `--seed` defaults to
12345, with a separate seed per trial. Warmup uses a separate sequence. The JSON
stores per-call changed fit-coordinate counts, likelihood values, sequence hashes
and timing arrays. Parameter generation/setters are timed separately; the
reported evaluation duration starts after those setters and includes the actual
likelihood evaluation. Case JSON files and summaries report the cache, reweight,
histogram and full-evaluation times separately.

The constant case reuses dial responses but still recomputes event weights and
histograms. Global changes can touch the complete dial-response cache, exposing
costs absent from constant-point evaluations and potentially changing locality
for the subsequent weight stage. Compare both reweight and complete evaluation
times. No optimal factor is selected or installed automatically by the benchmark.

`--trials` defaults to two and `--rounds` to one. The shorter defaults give
2,560 timed evaluations across all four workloads/layouts for one thread count,
including 160 global evaluations. The script prints the total call budget before
loading. Increasing `--calls` retains the global cap unless `--global-calls` is
also increased. These are short diagnostics: 100-call local sweeps cover only
part of a large parameter list.

The [CCLyon result analysis](CpuWeightBenchmarkCCLyon20261020.md) explains the
revised budgets: the earlier 96,000-evaluation run spent 74 minutes on global
evaluations alone and does not establish an advantage for factors above two.

## Validation and limits

The C++ regression compares weights, histogram sums and sums of squares **bit for
bit** between Fixed and Dynamic on the same input model. Coverage includes cached
and mixed dials, caps, external weight changes, reordered metadata, zero/negative/
nonfinite values, 1/2/5 threads, workloads too small for the queue, uneven tails
and an event dominating estimated work. The profiling regression verifies that
the default four-thread execution creates eight nonempty contiguous blocks,
processes every event, and exports correct worker counts. The Asimov Hesse
regression exercises the automatic Dynamic path.

Local numerical tests validate correctness and edge cases. The Mac input subsets
and hardware differ from the full-scale cluster model, so their throughput is
not evidence for the latter's speedup or regression. Earlier measurements with
4096/16384-event blocks evaluated a superseded prototype and do not measure the
current two-blocks-per-worker implementation. Full-scale performance still needs
a paired run on the same cluster allocation.

The revised Release implementation passes 107 selected C++ tests (excluding the
previously failing RootReaderCleanup configuration fixture) and all 29 Python
scripts numbered 210–232, including the Hesse and profile-coverage regressions.

Runtime-switching validation covers factors 1/2/5/10 and repeated switches back
to Fixed on the same loaded model, with bitwise prediction comparisons and model
storage identity checks. A small end-to-end benchmark verifies sixteen cases
(two thread counts, two rounds) with only two model loads and visible progress.

`233CpuWeightBenchmarkScenarios.py` runs all four workloads with original and
eigen coordinates, including a bounded parameter and an original-space fixed
parameter. It verifies 64 cases on two loaded models, intended coordinate-change
counts, sequence replay, likelihood and histogram agreement across layouts,
nonconstant predictions for moving workloads, and retained-model process IDs.

## Response-cache comparison

The same script accepts `--benchmark cache` to compare Indexed and Auto cache
traversal while keeping weight scheduling at Dynamic x2. See
[CPU cache traversal](CpuCacheTraversal.md) for the short paired command and
phase timing fields. The default `--benchmark weights` retains the weight-factor
comparison, using Auto cache traversal in all cases.
