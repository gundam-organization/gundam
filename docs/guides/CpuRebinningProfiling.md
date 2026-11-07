# CPU rebinning: inspection and benchmark protocol

Measured on 2026-10-23 using `upgrade-cc-2026`, four CPU threads, Release
(`-O3 -DNDEBUG`), ROOT 6.40.04 and Python 3.15. The model has 277,534 events,
1,778,971 unique dials and 374 histogram bins. Reference source: `92b1f80e`.

## Reference execution path

`CpuBackend::requestPropagation()` updates cached dial responses and event
weights, calls `updateEventState()`, then fills histograms and computes the
likelihood. The variable response composes a reference/override value with
multiplicative and additive dials; `pmuScale` therefore rescales the reference
`Pmu`, rather than accumulating changes between calls.

`EventStateCache` tracks inputs of variable dials. If those inputs are unchanged
and there are no host variable inputs or forced updates, both event-state jobs
are skipped. Otherwise, two separate jobs partition the whole event list into
fixed ranges: one recomputes variable responses, the other processes invalid
bin assignments. Variable input invalidation is global: changing any variable
dial triggers reevaluation of all variable responses in the event pass.

For strict binning, the CPU first checks whether the previous bin still contains
the event, provided reuse was certified safe at initialization. Otherwise
`BinningIndex` uses a complete rectangular grid where possible, or a spatial
tree followed by candidate sorting and exact interval checks. Smoothed binning
uses fractional contributions and can populate multiple bins.

The three supplied sample definitions are strict rectangular grids, with 165
(TPC), 170 (HAT), and 39 (SFG) bins. Static inspection of their text files confirms
complete grids, so grid lookup is expected unless another override changes the
effective geometry. This needs confirmation against the loaded model.

## Candidates to investigate

- Two full event traversals and two worker synchronizations whenever the
  event-state cache requires updating; fixed ranges may distribute unequal work.
- Variable responses are recomputed globally when a variable input changes.
- `updateEventState()` subsequently scans every event's contribution vector
  sequentially to count contributions. This falls outside the variable and
  binning stage timers, so their sum does not cover all event-state work.
- Strict binning still maintains a vector of contributions per event, and the
  histogram pass reads those vectors. This adds indirections and memory traffic;
  steady-state allocations are not assumed, since vector capacity is reused.
- When previous-bin reuse fails, strict candidates have their intervals checked
  for the hard bin and again by `BinningSemantics::fraction()`. The MPS interval
  pruning work is relevant here, though its benefit on the CPU remains unmeasured.

These were hypotheses from source inspection. Relevant
sources: `CpuBackendBinning.cpp`, `CpuBackend.cpp`, `EventStateCache.cpp`,
`BinningGeometry.cpp`, and `MpsEventPropagationKernels.metal`.

## Resident benchmark

`tests/benchmarks/226CpuRebinningProfiling.py` loads the production Asimov model
once per invocation, preserves dataset selections and backend settings, and
disables automatic output materialization during evaluations. It compares:

- `constant`: repeated identical parameter inputs, to measure cache reuse;
- `variable`: only `pmuScale` changes, cycling through -1, +1, -0.5, +0.5 prior
  standard deviations relative to its initialized value.

This gives scales 0.99/1.01/0.995/1.005 for the realistic case and
0.5/1.5/0.75/1.25 for stress. Identical absolute offsets can also be supplied
after conversion to sigma units, to separate migration amplitude from changes
in fit behavior. The script validates parameter presence, limits and finite
likelihoods. It exports raw per-call times, counters, likelihoods, final
histograms, parameter points, resolved configuration and median/p95 summaries.
It does not establish correctness against the legacy propagation path.

After the timed passes it explicitly materializes each parameter point, recording
the complete histogram, a digest of event bin assignments and the actual number
of migrations. These diagnostic event walks are excluded from timings.

Commands (run sequentially, with distinct output directories):

```bash
cd /Users/nadrino/Documents/Work/Output/results/gundam/upgrade-cc-2026
source ./env.sh
export PYTHONPATH="${GUNDAM_LIB_PATH}${PYTHONPATH:+:$PYTHONPATH}"
benchmarkScript=/Users/nadrino/Documents/Work/Repositories/gundam/tests/benchmarks/226CpuRebinningProfiling.py

python/venv/bin/python "$benchmarkScript" \
  --config ndUpCcInc.yaml \
  --override override/datasetHighlandv5.27.1_CCLyon.yaml \
    override/pmuScaleBinningTest.yaml override/backendCpuTest.yaml \
  --threads 4 --output outputs/cpu-rebin-realistic-t4

python/venv/bin/python "$benchmarkScript" \
  --config ndUpCcInc.yaml \
  --override override/datasetHighlandv5.27.1_CCLyon.yaml \
    override/pmuScaleBinningTest.yaml override/pmuScaleBinningStressTest.yaml \
    override/backendCpuTest.yaml \
  --threads 4 --output outputs/cpu-rebin-stress-t4
```

The stress override only changes the uncertainty and limits: it must follow the
base `pmuScaleBinningTest.yaml`, which defines the parameter and its variable dial
and disables the external Eb migration parameter set. Using stress alone does
not construct the intended workload.

On macOS, an installed directory in `DYLD_LIBRARY_PATH` can override even the
explicit extension path selected by `PYTHONPATH`. During development, select
the build's `src/PythonInterface` directory and remove the installed GUNDAM
directory from `DYLD_LIBRARY_PATH`. For the measurements here:

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/root/6.40.04_1/lib/root
export PYTHONPATH=/Users/nadrino/Documents/Work/Repositories/gundam/cmake-build-release/src/PythonInterface
```

`runtime.json` records the module hash and the actual dyld image paths. The
benchmark fails if macOS loads a different module image. Earlier `ab1` runs
were affected by this environment override and are excluded from comparisons.
`ab2` used the correct module but an intermediate implementation; `ab3` is the
final comparison. The original statically linked module/executable are retained
under `cmake-build-release/performance-baselines/cpu-rebin-20261023`.

First collect the four-thread baseline, then optionally repeat with 1/2/8 threads
and unique output directories. Compare variable/binning/histogram durations and
total evaluation time. `rebinnedEventCount` counts events processed by binning,
including successful previous-bin checks; it does not count actual migrations.
`binCandidateCount` includes previous-bin tests. Contribution counts also do not
measure how often events change bins. Native sampling or additional counters
may be needed to resolve the remaining costs and worker imbalance.

## Measured changes

The strict-binning path still tests the previous bin first, with the same
overlap certification and nonfinite-value fallback. It now uses a simple strict
membership check and leaves the existing contribution untouched on a hit.
After a miss, strict membership is evaluated once rather than also computing
a fractional response. Strict histogram filling reads the cached bin directly;
smoothed histograms retain the fractional contribution path.

Contribution counting is integrated into the worker pass, with a local counter
and a final reduction, removing the separate serial scan over every event's
contribution vector. In these measurements, worker scratch data was aligned to
128 bytes to keep hot counters and vectors apart on this machine's 128-byte cache lines. An intermediate
version added a counter without preserving worker isolation and regressed;
that implementation was discarded.

The subsequent portability change centralizes alignment in `CpuCacheAlignment.h`:
`cpuDestructiveInterferenceSize` uses `std::hardware_destructive_interference_size`
when available, with documented fallbacks for older standard libraries. All CPU
worker records and the scheduling counter use this named separation. The current
Apple toolchain recommends 256 bytes; this is not the physical cache-line size.
The Release build and 147 C++ tests pass with this change; the timing tables below
remain the measurements of the preceding 128-byte implementation.

Final measurements: 8 warmup evaluations and 3 trials of 60 evaluations per
scenario/variant, alternating the before/after order between realistic and stress
cases. Each process loads the full input once; no fit, Hesse or event-state host
copy is included in these timings. Numbers are medians in milliseconds.

| Workload | Actual migrations per change | Binning before → after | Evaluation before → after | Evaluation time reduction |
| --- | --- | --- | --- | --- |
| Realistic, ±1/±0.5 sigma | 3.07–6.18% | 0.653 → 0.535 | 2.134 → 1.947 | 8.8% |
| Stress, ±1/±0.5 sigma | 73.98–89.26% | 2.314 → 2.091 | 3.808 → 3.467 | 8.9% |
| Small, ±0.01/±0.005 sigma | 0.032–0.066% | 0.489 → 0.331 | 1.950 → 1.665 | 14.6% |

The small case uses the realistic prior and `--sigma-offsets -0.01 0.01 -0.005 0.005`,
so the absolute scale moves by 0.00005–0.0001. This is the most favorable case
for previous-bin reuse. Its binning time decreases by 32.3% (1.48x throughput).
Histogram filling in the variable cases decreases from roughly 0.152–0.155 ms
to 0.117–0.121 ms. Weight evaluation remains around 0.72–0.76 ms and is now a
larger fraction of the realistic evaluation cost. The stress case still spends
most of its time looking up migrated events.

For all three cases, all 1,080 measured likelihood values and associated counters
match exactly across variants. The final constant/variable histograms and the
five explicit diagnostic points per case (histograms, likelihoods and bin
digests) also match exactly. Existing C++/Python tests compare the backend with
the legacy path, including smoothing, overlaps, boundary points, nonfinite values,
host edits, invalidation and retry after failure.

Artifacts are in the input directory's `outputs/`:

- `cpu-rebin-ab3-{baseline,optimized}-{realistic,stress,small}-t4/`
- `cpu-rebin-comparison.json` (exact comparisons, timings and migration fractions)
- `cpu-rebin-fit-{baseline,optimized}-{realistic,stress}.{root,log}`
- `cpu-rebin-full-fit-comparison.json` (fit statistics, errors and covariances)

The local results measure these inputs on this machine; they do not establish
scaling on a cluster or predict a proportional speedup of an entire fit.

## Full fitter runs

Run the realistic command from the request unchanged for the initial end-to-end
baseline, using a distinct `-o` output for each repetition. For stress insert
`override/pmuScaleBinningTest.yaml` immediately before the stress override. Keep
Hesse for this baseline; Migrad and Hesse can exercise different patterns of
parameter changes. Record convergence/status and likelihood as well as runtime.

For a separate instrumented fit, add `resources/examples/CpuProfiling.yaml`
(via an absolute path) last among the overrides, with a distinct profile output
path per run. Its existing JSONL export includes variable and binning times, but
worker probes cover cache/weights/histograms only, and the rebinning counters
are currently available through `BackendTimingSummary`, not that JSONL export.
The ordinary full-fit baseline should be kept separate from this probe run.

Both full fits were run before and after the change, including Hesse. Migrad and
Hesse return status 0 and covariance status 3 in all runs. Fit parameter values,
likelihoods, non-timing statistics and all 44 checked post-fit error/covariance
histograms per configuration match exactly. Migrad call counts remain 51,106
(realistic) and 51,100 (stress); Hesse counts remain 49,596 and 49,592.

Observed wall times are 158.29 → 151.39 s (realistic) and 155.92 → 150.73 s
(stress). These are single end-to-end runs; reference runs overlapped briefly
with compilation/tests, so the wall times are indicative, not a controlled
end-to-end speedup claim. Accumulated histogram time drops from about 20.5 s
to 14.1 s across roughly 102,000 backend calls. These fits spend most of their
time changing weight parameters; they do not sustain large momentum migrations
like the targeted stress benchmark.

`231MpsHessePrecision.py` checks MPS numerical curvature/Hesse precision; it is
not a rebinning benchmark. For later CPU changes, the relevant existing checks
are `217DynamicRebinning.py` and `220CpuDynamicBinning.py`, followed by comparisons
of likelihoods and histograms at identical parameter points before/after changes.

Validation of the final build: 147 C++ tests pass when excluding the two
`RootReaderCleanup` cases. The complete suite exits during their configuration
because a structured file entry lacks mandatory `name`; the same configuration
failure was reproduced with the unchanged baseline module. No changes were made
to those unrelated tests. Python tests 217, 219, 220 and 231 pass with the corrected
loader environment. Logs and generated test artifacts are retained under
`cmake-build-release/cpu-rebin-python-tests` (C++ logs under `/tmp/gundam-cpu-rebin-*`).
