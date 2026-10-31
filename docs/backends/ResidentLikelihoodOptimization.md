# MPS statistical likelihood optimization — 30 October 2026

## Retained implementation

The optimized MPS backend now uses one likelihood bin per thread for up to
8,192 bins, and chunks of eight above that size. Both host allocation/dispatch
and the Metal shader read the same `LikelihoodWorkLayout`. Final partials are
reduced by a deterministic 128-thread tree, replacing the serial final sum.
No atomics, reduced-precision arithmetic or changes to the per-bin formulas were
introduced. The validated JointProbability classes remain untouched.

Resident histograms and likelihood are reused when the previous result is valid
and no relevant weight dependency or observable/binning state changed. External
weight generations, observable revisions, host edits and forced binning updates
prevent reuse. Recompute mode continues to recompute. The backend still reads
one 8-byte likelihood total per evaluation; it does not materialize weights or
histograms during minimization.

The existing `cachedResponses` default is preserved. CPU and CUDA execution
algorithms are unchanged. The new work descriptor is independent of the
likelihood configuration; there is no new user-facing configuration key.

## Validation

- Full Release build passes.
- **244 fast C++ tests pass**, including ten dedicated likelihood tests.
- All three BackendFitCovariance tests pass, including original-engine versus
  MPS fits, parameter errors and Hesse covariance for Poisson and Barlow 2022.
- New device tests cover reduction sizes 1, 127, 129, 4,140, 8,192, 8,193 and
  65,536, including ignored bins and uneven work counts.
- Cache tests verify reuse for an irrelevant parameter change and invalidation
  by an external-weight generation or explicit forced binning update. Existing
  observable/binning tests also pass.
- A dedicated Barlow device test uses rates from **1e-12 to 1e8**, relative
  prediction shifts of **0, ±1e-8 and ±1e-5**, and several fractional variances.
  It compares to the original class using a relative LLH tolerance rather than
  an absolute tolerance that would hide errors at tiny rates. All cases pass.
- Production benchmarks assert identical seeded coordinate digests across CPU
  and MPS, finite likelihoods and scalar-only MPS readback at every timed point.

Metal still has float32 exponent limits despite compensated precision. These
checks do not extend its exponent range or establish binary64 accuracy over
all extreme inputs. Separate production fits to convergence were not run.

## Production benchmark protocol

Apple M4 Max, 48 GiB RAM, macOS 27.0.1, Release build, four CPU threads.
Three trials rotate CPU, cached MPS and recompute MPS order on a single loaded
Asimov engine per run. Ten warmup calls per pass. Parameter setters, backend
construction and histogram materialization are outside evaluation timing.
Local scenarios cover every effective fit coordinate, with seeded 0.1-sigma
in-domain displacements. PCA, plots, step-response initialization, minimization
and Hesse are disabled for performance measurements.

All tables show the median of three trial medians, in **milliseconds per full
likelihood evaluation**, including the ordinary Python/C++ call and parameter
constraints. These are evaluation workloads, not end-to-end fit speedups.
Stage probes use separate synchronized commands and are not added to ordinary
evaluation timings. No hardware occupancy/bandwidth counters were collected.

The runs took place on an interactive desktop without affinity/clock pinning.
The initial OA2024 baseline preceded an overnight pause; final measurements
were taken after resuming. CPU control timings are therefore included below.
Per-trial distributions and p95 samples are retained in the raw results.

## OA2024: 4,140 bins

Inputs: `/Users/nadrino/Documents/Work/Output/results/gundam/GundamInputOA2024`.
The available configuration is `configOA2024.yaml`; the benchmark uses
`override/v13ProdFewRuns.yaml`, preserving its complete selected files.
The supplied `ndUpCcInc.yaml`/Highland command names refer to the Upgrade
configuration, measured separately below. No event-count reduction was applied.

The loaded OA2024 model has **1,601,913 events**, 4,140 bins and 842 effective
fit coordinates. The benchmark reads legacy frontend/likelihood options without
rewriting the user's input files. Before/after resolved configurations,
parameter manifests and every seeded sequence digest are identical.

Barlow-Beeston 2022, cached MPS:

| Scenario | Before | Optimized | Time reduction |
|---|---:|---:|---:|
| constant | 0.644 | 0.284 | 55.9% |
| local1 | 1.721 | 1.321 | 23.3% |
| local2 | 1.724 | 1.323 | 23.3% |
| global | 7.931 | 7.646 | 3.6% |

CPU controls, unchanged implementation:

| Scenario | Before | Final run |
|---|---:|---:|
| constant | 5.465 | 5.547 |
| local1 | 5.499 | 5.491 |
| local2 | 5.336 | 5.258 |
| global | 29.633 | 31.540 |

The local-evaluation improvement is clear with nearly identical CPU controls.
The global-evaluation change is modest and should not be interpreted as a
universal speedup: dense dial evaluation dominates there, and desktop variability
is visible in the CPU control.

Maximum statistical likelihood difference before/after at identical points:
**2.09e-11**.
Maximum final MPS-versus-CPU difference across all trials is below **1.3e-10**.

An intermediate run used eight bins/thread at 4,140 bins and gave about 1.38 ms
for local1. The isolated parallel-reduction microbenchmark then favored one
bin/thread at this size, motivating the final 8,192-bin threshold. The final
local1 median is about 1.32 ms. The initial threshold is not retained.

## Upgrade CC 2026: 374 bins

Configuration `upgrade-cc-2026/ndUpCcInc.yaml` plus
`override/datasetHighlandv5.27.1_CCLyon.yaml`: 277,177 events, 374 bins and
313 effective fit coordinates. Barlow retains `BBNoUpdateWeights: true`.
This benchmark uses the same one-bin-per-thread path as the final build; its
module was compiled before the threshold was extended from 4,096 to 8,192,
which does not change the 374-bin execution plan. Poisson below uses the final
module directly. Module hashes are recorded in every model manifest.

Barlow-Beeston 2022, cached MPS:

| Scenario | Before | Optimized | Time reduction |
|---|---:|---:|---:|
| constant | 0.363 | 0.148 | 59.3% |
| local1 | 0.570 | 0.323 | 43.3% |
| local2 | 0.571 | 0.317 | 44.5% |
| global | 1.599 | 1.180 | 26.2% |

Maximum before/after statistical LLH difference:
**2.17e-11**.

Poisson, cached MPS:

| Scenario | Before | Optimized | Time reduction |
|---|---:|---:|---:|
| constant | 0.269 | 0.144 | 46.6% |
| local1 | 0.444 | 0.322 | 27.4% |
| local2 | 0.443 | 0.326 | 26.4% |
| global | 1.406 | 1.167 | 17.0% |

Poisson enables `ignoreBinsWithZeroPredictionAtPrior` in both runs to avoid
infinite empty-prior-bin contributions. Maximum before/after statistical LLH
difference: **3.73e-11**.

## Isolated kernel scaling

The native microbenchmark compiles the actual generated shader and compares
chunk sizes on synthetic bins. Every total is checked against the portable
CPU evaluator. It records both per-evaluation submit/wait/readback latency and
batched GPU time; neither is a production evaluation time.

For Barlow, batched GPU time in microseconds:

| Bins | Previous serial finish, chunk 64 | Parallel finish, chunk 1 | Parallel finish, chunk 8 |
|---|---:|---:|---:|
| 374 | 335.7 | 12.9 | 60.0 |
| 4,140 | 401.6 | 17.2 | 67.1 |
| 65,536 | 578.1 | 116.6 | 76.6 |

This supports the retained small-/large-bin distinction and confirms that the
serial-final-reduction bottleneck at large partial counts has been removed.
The threshold is an empirical policy for these tested workloads, not a hardware-
independent optimum. No runtime tuning or precision change is involved.

## Artifacts and reproduction

[Machine-readable summary](../../cmake-build-release/benchmarks/LikelihoodOptimization-20261030/summary.json),
raw samples, configurations, parameter manifests, source patch, microbenchmark
results and test/build logs are under
`cmake-build-release/benchmarks/LikelihoodOptimization-20261030/` (ignored by git).
Intermediate eight-bin OA2024 results are excluded from the final tables.

```sh
cmake --build cmake-build-release -j3
env -u DYLD_LIBRARY_PATH \
  PYTHONPATH="$PWD/cmake-build-release/src/PythonInterface" \
  python3.15 tests/benchmarks/233ResidentLikelihoodBenchmark.py \
  --work-dir /Users/nadrino/Documents/Work/Output/results/gundam/GundamInputOA2024 \
  --config configOA2024.yaml --override override/v13ProdFewRuns.yaml \
  --calls 60 --global-calls 20 --probe-calls 4 \
  --output /tmp/new-oa2024-likelihood-benchmark
```

The native `234MetalLikelihoodMicrobenchmark.mm` contains its compilation
command and now configures the current parallel shader through the shared
work descriptor. Historical serial-shader timings remain archived rather
than silently being replaced by measurements of the new kernel.
