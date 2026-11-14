# OA2024 backend memory — 23 October 2026

The enlarged input selection in `GundamInputOA2024` contains **1,601,913 events,
13,623,174 unique dials and 25,284,953 event/dial references**, with 790 backend
parameters and 4,140 histogram bins. The selection includes the newly added run3
and run4a MC spline files. All selected files and events are retained.

## Changes retained

Three local changes in the common backend model builder reduce memory without
changing floating-point storage, dial indices or per-event multiplication order:

* Remove `EngineBindings::eventDials`, a per-occurrence pointer list with no
  consumer. Event bindings and event/dial indices already provide the links
  required by the backends.
* Store an input sequence once per source `DialInputBuffer`, instead of once per
  unique dial. Sharing uses pointer identity, preserving the entire sequence,
  parameter order and mirror settings. Different buffers remain separate.
* Count unique dials, payload values and weight references before filling their
  vectors, then reserve the required capacity. The same interface lookup table
  is reused for compilation, with a sentinel for interfaces not yet compiled.
  Indices retain their first-encounter order. No second large lookup table or
  payload copy is introduced.

These changes apply to the shared CPU/MPS model. CPU propagation kernels,
interleaving, scheduling and binning are unchanged. Existing rebuilds retain
vector capacity as before. Compacting the dial descriptor, changing numerical
precision or removing native dial ownership would require a broader change and
are outside this optimization.

## Memory measurements

Apple M4 Max, 48 GiB RAM, Release build, four threads. The reference Python
module was copied before editing the production sources. Each measurement runs
in a new process. The benchmark verifies the actual loaded module path and
records its SHA-256; `DYLD_LIBRARY_PATH` contains only the ROOT library directory.

Representative model buffers, decimal MB:

| Buffer | Used before → after | Capacity before → after |
| --- | ---: | ---: |
| Dial input descriptors | 435.942 → 0.025 | 536.871 → 0.033 |
| Dial descriptors | 1,852.752 → 1,852.752 | 2,281.701 → 1,852.752 |
| Spline/graph payload | 1,375.115 → 1,375.115 | 1,677.722 → 1,375.115 |
| Event/dial indices | 101.140 → 101.140 | 134.218 → 101.140 |

The number of input descriptors falls from **13,623,174 to 790**. The separate
unused pointer list is also removed; it is not part of the profiler's buffer
table. Cached responses and cache-group counts remain unchanged.

Memory immediately after initialization, decimal GB:

| Measurement | Reference 1 | Optimized 1 | Reference 2 | Optimized 2 | Optimized 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| malloc bytes in use | 11.290 | 9.703 | 11.290 | 9.703 | 9.703 |
| malloc reserved bytes | 22.748 | 21.148 | 23.201 | 21.169 | 21.149 |
| Current RSS | 18.542 | 16.083 | 16.484 | 16.862 | 16.516 |
| Peak RSS through initialization | 18.897 | 16.080 | 17.080 | 17.118 | 16.544 |

Live allocator storage decreases reproducibly by **1.587 GB (14.1%)**. RSS is
less stable: macOS page compression/residency and other desktop activity affect
it, and a lower peak RSS is not reproduced in every pair. A vector's reserved
capacity is not a measurement of resident pages. RSS and peak RSS come from
different OS queries, so their snapshot values can differ slightly.

## Timing and numerical validation

`228CpuMemoryBenchmark.py` uses the configuration and overrides from:

```sh
gundamFitter -c configOA2024.yaml \
  -of override/v13ProdFewRuns.yaml override/backendCpuTest.yaml -a -t 4
```

It initializes the full Asimov model without running a minimization or Hesse.
PCA/step-response initialization and plotting are disabled. Profiling exports
one initial record, with no sampled OS probes during timed calls. Parameter
setters and explicit histogram materialization are outside the timer.

Each process executes two trials per scenario, with five warmup evaluations.
Constant uses 60 calls per trial; local1 visits every effective coordinate;
global uses ten calls per trial. Seed 12345 and step scale 0.1 are fixed.

Median evaluation times in milliseconds:

| Scenario | Reference 1 | Optimized 1 | Reference 2 | Optimized 2 | Optimized 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant | 5.024 | 6.829 | 5.018 | 4.999 | 5.118 |
| One coordinate changes | 5.062 | 6.900 | 5.092 | 5.090 | 5.132 |
| All coordinates change | 27.258 | 29.421 | 25.746 | 25.913 | 27.695 |

The first optimized process has an unexplained histogram-stage increase
(approximately 2.54 ms versus 0.77 ms), without a corresponding weight-stage
increase. It is not reproduced by either subsequent optimized process. No mechanism
or general speed improvement is claimed. Initialization takes 126.6/133.7 s
for the references and 129.6/128.1/136.3 s for the optimized processes.

Parameter manifests, resolved model configurations, sequence digests, every
likelihood, and final histogram contents/errors match exactly against the
reference. The benchmark supports `--reference` to repeat these checks and save
`comparison.json`.

Validation of the shared model also includes:

* 149 native tests passed, including a new `EngineLayout` regression covering
  shared/different input buffers, mirroring, two-input surfaces, reversed input
  order, empty inputs/events and repeated model builds. It compares CPU weights
  against native dial evaluations.
* Ten Python tests passed: 211 CPU, 212 MPS, 220 CPU dynamic binning, 221 MPS
  dynamic binning, 227 shared spline loading, 228 shared binned dials, 230 Asimov
  Hesse stability, 231 MPS Hesse precision, 232 CPU profiling and 233 parameter
  proposal scenarios.
* The full native invocation stops in the existing `RootReaderCleanup` fixture:
  its structured `filePathList` entry lacks the mandatory `name` field. The
  149-test successful run excludes its two parameterized cases. This fixture is
  unchanged by the memory optimization.

Raw runs are retained under
`GundamInputOA2024/outputs/oa2024-memory-{baseline,optimized}-20261023` and
`oa2024-memory-{baseline,optimized}-repeat-20261023`, plus
`oa2024-memory-optimized-check-20261023` for the third optimized process.
Test logs are in `cmake-build-release/memory-validation/`.
The baseline module/executable are in
`cmake-build-release/performance-baselines/memory-oa2024-20261023/`.
