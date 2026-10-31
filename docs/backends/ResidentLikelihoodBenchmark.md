# CPU/MPS resident likelihood benchmark — 30 October 2026

This is the historical baseline and temporary chunk-size experiment. The subsequent
[retained optimization and validation](ResidentLikelihoodOptimization.md) supersedes
its production implementation description. Raw measurements below are unchanged.

## Scope and protocol

Apple M4 Max, 12 performance cores + 4 efficiency cores, 48 GiB RAM, macOS
27.0.1. Release build (`-O3`, git `0c047b06`), Python 3.15, freshly rebuilt
GUNDAM module. The restored baseline module SHA256 is
`f175fd4cba68455ee9a29de34d82e72bf40232581efd7c0526f4ab455d47a2c7`, identical
to the module used for the baseline measurements.

Production configuration: `upgrade-cc-2026/ndUpCcInc.yaml` with
`override/datasetHighlandv5.27.1_CCLyon.yaml`, under
`/Users/nadrino/Documents/Work/Output/results/gundam`. No event-count reduction.
277,177 events, 1,776,300 unique cached dials, 4,302,714 event-dial references
(as reported by the runtime index byte count), 374 bins, 313 fit coordinates.
The configured Barlow function uses `BBNoUpdateWeights: true`. Poisson is a
separate experiment with `ignoreBinsWithZeroPredictionAtPrior: true` so empty
prior bins do not make its total infinite. It retains the same events and bins;
likelihood values across the two statistical functions are not compared.

Each process loads the dataset once, then replaces the backend on the same
engine. Initialization/backend construction, parameter setters, diagnostics and
histogram materialization are outside the evaluation timer. Timed evaluations
use Asimov data, no minimization/Hesse/plots/PCA/step-response initialization,
and automatic output materialization disabled. Three trials rotate backend
order. Each pass has ten warmup evaluations. Every trial replays the same seeded
sequence on each backend, with displacements of 0.1 prior sigma and domain checks.

- `constant`: 100 repeated evaluations at the nominal point.
- `local1`: 313 evaluations, changing one coordinate per call and covering every coordinate.
- `local2`: 157 evaluations, changing two coordinates per call.
- `global`: 40 evaluations, changing all coordinates per call.

Tables report the median of three trial medians in **milliseconds per full
likelihood evaluation**, including the normal Python/C++ call boundary and
parameter-constraint evaluation. These are controlled workloads, not an actual
Migrad trajectory or an end-to-end fit speedup. Runs were sequential on an
interactive desktop; no CPU affinity, clock pinning or OS isolation was used.
Raw samples, trial medians and p95 values are preserved.

## Barlow-Beeston 2022

| Scenario | CPU 1 thread | CPU 4 threads | CPU 12 threads | MPS cached | MPS recompute |
|---|---:|---:|---:|---:|---:|
| constant | 2.408 | 0.699 | 0.336 | 0.363 | 1.341 |
| local1 | 2.432 | 0.708 | 0.374 | 0.570 | 1.428 |
| local2 | 2.439 | 0.712 | 0.371 | 0.571 | 1.428 |
| global | 12.835 | 3.572 | 2.199 | 1.599 | 1.608 |

At four CPU threads, cached MPS is about 1.24x faster for local changes and
2.23x faster for global changes. At twelve threads CPU is faster than the current
MPS implementation for local changes (0.374 vs 0.570 ms), while MPS remains
faster for global changes (1.599 vs 2.199 ms). Twelve threads is the fastest CPU
setting tested, not a claim of the optimal thread count.

Always recomputing MPS responses costs about 2.5x the cached MPS evaluation for
local changes. Both GPU modes are similar for global changes. Keep
`cachedResponses` for coordinate-style fits on this workload.

CPU four-thread stage medians: event weights about 569 us, histograms 113 us,
statistical likelihood 3.7 us for local1; dial-cache evaluation rises from about
9 us locally to 2,829 us globally. The statistical likelihood is not the CPU
optimization priority on this dataset. Timers overlap or have different scopes;
they are not a partition of every measured call.

## Poisson

| Scenario | CPU 4 threads | MPS cached | MPS recompute |
|---|---:|---:|---:|
| constant | 0.695 | 0.269 | 1.241 |
| local1 | 0.710 | 0.444 | 1.303 |
| local2 | 0.712 | 0.443 | 1.304 |
| global | 3.571 | 1.406 | 1.416 |

Poisson is cheaper on Metal than Barlow, while the CPU evaluation remains
dominated by propagation. CPU thread scaling was measured for Barlow only.

## Controlled Metal chunk-size experiment

The existing likelihood kernel gives each thread up to 64 bins. With only
374 bins, that produces six threads doing the expensive compensated arithmetic
serially. A temporary candidate changed both dispatch/allocation and the shader
to eight bins per thread. No arithmetic formula or precision change was made.
The seven likelihood tests passed with this candidate, including near-minimum
cases and scalar-only readback checks.

Full production evaluations, Barlow, MPS cached:

| Scenario | Baseline 64 bins/thread | Candidate 8 bins/thread | Time reduction |
|---|---:|---:|---:|
| constant | 0.363 | 0.234 | 35.6% |
| local1 | 0.570 | 0.369 | 35.2% |
| local2 | 0.571 | 0.369 | 35.4% |
| global | 1.599 | 1.231 | 23.0% |

The CPU control measured in the candidate pass remains close to the baseline
(0.706 vs 0.708 ms local1). MPS/CPU statistical LLH differences remain below
5e-11 at every timed point, in both baseline and candidate experiments. This
validates ordinary point values for this experiment; it does not replace a
full independent precision study or toy-fit convergence campaign.

The candidate was **not retained**. Tracked backend sources, generated Metal
source and the Release module were restored. The restored module hash matches
the baseline exactly, and all seven likelihood tests pass again. The precise
candidate patch is saved with the raw results.

An isolated microbenchmark of the actual generated Metal source separately
compares 64, 8 and 1 bins/thread on 374, 4,140 and 65,536 synthetic bins. It
checks totals against the double-precision CPU evaluator. It runs both forward
and reverse candidate orders. For Barlow, median single-evaluation latency in
microseconds (first run; reverse run gives the same main conclusion):

| Bins | 64 bins/thread | 8 bins/thread | 1 bin/thread |
|---|---:|---:|---:|
| 374 | 491 | 209 | 184 |
| 4,140 | 564 | 253 | 461 |
| 65,536 | 785 | 888 | 5,011 |

These timings include submission/wait and the scalar readback for an isolated
likelihood command, not production propagation. Batched GPU timings are also
saved. A universal switch to one bin per thread is unsuitable: the existing
final kernel serially sums the partials, becoming expensive as their count grows.

## Optimization priorities

1. **Adapt the likelihood work distribution to bin count.** Eight bins/thread
   is a measured improvement on the production model, but large-bin benchmarks
   favor a larger chunk with the current final reduction. A parallel tree/SIMD
   reduction could remove that restriction. This is the strongest measured
   optimization opportunity.
2. **Keep cached responses for local changes.** This is already configurable;
   recompute mode is substantially slower for this workload.
3. **Avoid histogram/LLH work when the prediction has not changed.** The current
   MPS executor skips event-weight updates for constant inputs but still reduces
   histograms and evaluates the statistical LLH. Cache/revision-based reuse is
   a plausible additional improvement; it was not implemented or timed here.
   It must account for external-weight generations and dynamic observables.
4. **Keep device command work together.** The ordinary local1 MPS measurement
   has about 396 us of GPU time and 545 us of CPU wait, which overlap. Command
   encoding itself is about 7 us. Synchronization/submission overhead remains,
   but per-stage probes submit separate commands and cannot establish an exact
   removable overhead budget. Avoid adding synchronizations.
5. **On CPU, focus on propagation and scaling, not the LLH formula.** The current
   LLH is only a few microseconds. Twelve threads improves this model over four;
   memory traffic and dense dial evaluation warrant further profiling before
   changing kernels.

No hardware occupancy, bandwidth or energy counters were collected. Separate
Metal stage probes are diagnostic and run with extra command submissions and
synchronizations; their durations must not be added to ordinary wall timings.

## Artifacts and reproduction

[Machine-readable summary](../../cmake-build-release/benchmarks/ResidentLikelihood-20261030/summary.json)
and full samples/configurations/logs are under
`cmake-build-release/benchmarks/ResidentLikelihood-20261030/` (ignored by git).
The incomplete first pilot run was excluded from the summary.

```sh
env -u DYLD_LIBRARY_PATH \
  PYTHONPATH="$PWD/cmake-build-release/src/PythonInterface" \
  /opt/homebrew/bin/python3.15 -u tests/benchmarks/233ResidentLikelihoodBenchmark.py \
  --work-dir /Users/nadrino/Documents/Work/Output/results/gundam/upgrade-cc-2026 \
  --config ndUpCcInc.yaml \
  --override override/datasetHighlandv5.27.1_CCLyon.yaml \
  --output /tmp/new-resident-likelihood-benchmark
```

Use `--likelihood PoissonLLH` for the Poisson comparison, or
`--threads 12 --cpu-only --probe-calls 0` for CPU scaling. The native
`234MetalLikelihoodMicrobenchmark.mm` documents its compilation command; an
optional `--reverse` argument reverses chunk order. It only modifies source
strings in its own process and leaves production code unchanged.
