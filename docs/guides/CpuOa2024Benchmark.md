# OA2024 CPU propagation benchmark — 23 October 2026

Inputs: `/Users/nadrino/Documents/Work/Output/results/gundam/GundamInputOA2024`.
Reference revision: `95fac21e`. Release build (`-O3 -DNDEBUG`), Apple LLVM 21,
four resident CPU threads, 48 GiB physical RAM. All files selected by
`v13ProdFewRuns.yaml` are retained; there is no event-count reduction.

## Production-command profiling

The supplied command was launched with an isolated ROOT/log output:

```bash
gundamFitter -c configOA2024.yaml \
  -of override/v13ProdFewRuns.yaml override/backendCpuTest.yaml -a -t 4
```

The CPU override already enables sampled profiling (one sample every 100 calls,
maximum 200 samples). The model contains:

| Quantity | Value |
| --- | ---: |
| Events | 996,951 |
| Unique cached dials | 8,263,097 |
| Event/dial references | 15,439,544 |
| Backend input parameters | 790 |
| Effective fit coordinates in the benchmark | 842 |
| Histogram bins | 4,140 |
| Shared input-transform groups | 790 |

Dynamic x2 scheduling, Auto cache traversal and Shared input evaluation were
already enabled. Event binning is static for this model. Large resident arrays
include about 1.12 GB of dial descriptors, 843 MB of used spline payloads and
66 MB of cached responses. The initialization log reports roughly 7.5 GB of
allocated engine memory, with peak process RSS around 13 GB during loading.

The first 195 post-startup samples, through call 19,500, have median stage times
of 2.815 ms for event weights, 0.487 ms for histograms, 0.026 ms for the dial cache
and 0.070 ms for the statistical likelihood. Dense initial cache updates are much
more expensive. These measurements motivated testing the event-weight kernel.

The profiling fit was paused during isolated resident benchmarks and resumed
after each comparison. It was then deliberately interrupted before convergence;
the last logged Migrad counter is 65,981. Neither Migrad nor Hesse completion is
claimed for this production configuration. Pauses contaminate the enclosing fit
and stage timers, so **no end-to-end fit speedup is inferred** from this run. The
ROOT output is named `.partial.root`. Profiling JSONL from the initial sampling
window and pause timestamps are retained separately.

## Optimization retained

The previous kernel finishes one event's product before beginning the next.
`CpuWeightKernels::reweightInterleaved()` advances two independent products
together, through the common prefix of their dial counts, then handles their
remaining dials and any odd final event. This gives the processor independent
response loads/products to overlap. No hardware-counter evidence is claimed for
the precise microarchitectural mechanism.

For each event, the dial multiplication order, double precision, cap placement
and final multiplication by the base weight remain unchanged. No event/dial or
payload buffers are copied or reordered. The interleaved path is retained as
the default when every referenced response is cached. Mixed cached/external
evaluation uses the existing scalar kernel, preserving evaluator call order.

The scheduling policy, block factor, cache traversal and input-sharing defaults
are unchanged. The preliminary Fixed/Dynamic comparison showed only a small
scheduling difference on this model, insufficient to justify another policy.

`BackendManager.setDiagnosticCpuWeightInterleaving(bool)` is a synchronous
diagnostic switch for the original/interleaved kernels. It allows A/B comparisons
on one resident model without changing its allocation or rebuilding its caches.

## Controlled A/B measurements

`tests/benchmarks/227CpuWeightInterleavingBenchmark.py` loads the full Asimov
model once. It disables sampled OS probes and output materialization during
timed evaluations, then replays identical seeded sequences of effective fit
coordinates. PCA/step-response initialization and plotting are disabled as in the
existing CPU resident benchmarks. This is a controlled propagation workload,
not a replay of Migrad's exact trajectory.

The benchmark uses two rounds and two trials per case, five warmup calls per
pass, and both serial-first and interleaved-first orders. Local1 covers all
842 coordinates, Local2 uses 421 calls to cover them in pairs. Constant uses
60 calls and Global uses 10 per trial. Parameter setters and final histogram
materialization are outside the evaluation timer.

The table gives medians of the four trial medians, in milliseconds, with four
active weight workers and the resident four-thread pool:

| Scenario | Weight stage, serial → interleaved | Full evaluation, serial → interleaved | Evaluation time reduction |
| --- | --- | --- | ---: |
| Constant parameters | 2.769 → 2.338 | 3.472 → 3.031 | 12.7% |
| One coordinate changes | 2.787 → 2.348 | 3.520 → 3.079 | 12.5% |
| Two coordinates change | 2.788 → 2.356 | 3.528 → 3.097 | 12.2% |
| All coordinates change | 2.998 → 2.478 | 16.140 → 15.508 | 3.9% |

The weight-stage reduction is about 15–17%; its throughput improves by about
18–21%. Global evaluation gains are smaller because dial-response updates
dominate that workload. This does not establish a whole-fit speedup or a speedup
on other processor architectures.

The same process also tests one active **weight worker**, retaining four cache
and histogram workers. Local evaluation times decrease from about 10.94 to
9.34 ms. This control is not a single-thread whole-backend benchmark.

There are 32 completed A/B pairs and 21,328 timed evaluations. Independent
checks confirm exact agreement for all 10,664 corresponding likelihood values,
all parameter-sequence digests and final histograms. No numerical tolerance is
needed for the interleaving comparison.

## Validation and second-model control

- 148 C++ tests pass, excluding the two previously identified `RootReaderCleanup`
  configuration failures. The new bitwise kernel regression covers zero/unequal
  dial counts, odd/even and empty ranges, capped/uncapped products, negative
  weights, NaN/Inf and the mixed-evaluator fallback.
- Python regressions 211, 220, 230, 231, 232 and 233 pass, including analytic
  Asimov curvature/Hesse checks for CPU and MPS, dynamic binning and profiling.
- A paired control on `upgrade-cc-2026`, with the Pmu scale/rebinning override,
  completes 16 A/B pairs with exact likelihood/histogram equality. Local evaluation
  times decrease from about 1.081 to 0.932 ms (13.7%); global times decrease from
  4.632 to 4.528 ms (2.3%). No performance regression is observed in this control.

## Reproduction and artifacts

Use the matching Python 3.15 environment and select the rebuilt extension.
On macOS, exclude the installed GUNDAM directory from `DYLD_LIBRARY_PATH` so
dyld cannot substitute a different module. The benchmark records actual loaded
image paths and rejects such a substitution.

```bash
cd /Users/nadrino/Documents/Work/Output/results/gundam/GundamInputOA2024
export OA_INPUT_FOLDER=/Users/nadrino/Documents/Work/Output/results/gundam/common
export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/root/6.40.04_1/lib/root
export PYTHONPATH=/Users/nadrino/Documents/Work/Repositories/gundam/cmake-build-release/src/PythonInterface

/Users/nadrino/Documents/Work/Output/results/gundam/upgrade-cc-2026/python/venv/bin/python \
  /Users/nadrino/Documents/Work/Repositories/gundam/tests/benchmarks/227CpuWeightInterleavingBenchmark.py \
  --config configOA2024.yaml --threads 4 --workers 4 1 \
  --output outputs/oa2024-cpu-interleaving-new-run \
  --override override/v13ProdFewRuns.yaml override/backendCpuTest.yaml
```

In the OA2024 input directory, `outputs/oa2024-cpu-interleaving-ab-20261023/`
contains settings, runtime images, parameter/model metadata, every raw timed
pass, `summary.json`, `aggregate.json` and `validation.json`. The preliminary
separate-process comparisons are in `outputs/oa2024-cpu-kernel-{baseline,paired}-20261023/`.
The reference profiling log, partial ROOT file and pause records use the
`outputs/oa2024-cpu-baseline-20261023` prefix. Its sampled profile is
`cpu-profile-4-41052.jsonl`.

The Upgrade control is in that input directory's
`outputs/oa2024-interleaving-upgrade-control-20261023/`. C++ and Python validation
logs are retained under `/tmp/gundam-oa2024-*` and
`cmake-build-release/oa2024-validation/`. The original executable/module and
source snapshots are retained under
`cmake-build-release/performance-baselines/oa2024-20261023/`.
