# CPU response-cache traversal

The CPU backend uses an automatic traversal policy when updating cached dial
responses. Event-weight scheduling remains Dynamic with two blocks per worker.
There is no new numeric YAML tuning option.

## Execution policy

The existing parameter-to-dial index lists determine the exact candidate count.
A small per-parameter byte mask records which raw parameters changed. If at
least half of all dial descriptors are candidates, the backend scans descriptor
indices in increasing order, skipping uncached dials, immutable shifts and dials
whose input parameter did not change. Otherwise, it retains the targeted
parameter-list traversal. A change to one parameter can select either path,
depending on how many dials it affects.

The linear path avoids constructing and copying the concatenated candidate-index
array. Descriptor and response accesses follow increasing indices, favoring
locality; actual payload access costs and load balance still depend on the model.
The targeted path remains available for sparse updates and models dominated by
external weights or constants. If no candidate exists, no cache worker job runs.
The half-descriptor criterion is an initial automatic policy to validate with
measurements, not a guarantee of improvement for every dense workload.

Dial math, input mirroring, transformed-input equality checks, extrapolation,
response limits/smoothing and double precision are unchanged. Workers write the
same unique response slots; per-event multiplication and histogram reduction
orders are unchanged. Immutable shift responses retain their build-time cache.
No descriptor, spline payload or response buffer is duplicated. Extra retained
storage is one byte per physical input parameter, plus scalar accounting.
Existing candidate storage may still grow on sparse updates; switching diagnostic
modes retains its capacity rather than freeing and reallocating it between cases.

## Short paired benchmark

Rebuild GUNDAM with its Python interface and select the rebuilt module. From the
input-model directory, run:

```sh
python3 /path/to/gundam/tests/benchmarks/222CpuWeightSchedulingBenchmark.py \
  --benchmark cache --config configOA2024.yaml --threads 8 \
  --calls 100 --global-calls 10 --warmup 3 --trials 2 --rounds 2 \
  --override override/v12ProdRun45.yaml override/backendCpuTest.yaml
```

Use the overrides for the intended dataset. This comparison fixes event-weight
execution to Dynamic x2, and compares:

- `Indexed`: the previous parameter-list order and concatenated indices.
- `Auto`: the new automatic linear/targeted selection.

All constant, local1, local2 and global workloads are included. There is one
model load for eight threads, followed by sixteen cases with reversed case
order in round two. The command performs **2,480 timed evaluations, including
80 global evaluations**, plus warmup and cache priming. This is 300 times fewer
timed global evaluations than the earlier 24,000-global-call run. Loading can
still take several minutes; it is excluded from timing. For a quicker first
check, use one round, giving 1,240 timed evaluations and 40 global ones.

The main log prints the exact call budget, phases and progress, with a process
heartbeat during quiet loading. The timestamp/PID output directory contains
native logs, settings, parameter manifests, individual case JSON and a summary.
Trial checkpoints and completed cases survive interruption as described in
[CPU weight scheduling](CpuWeightScheduling.md).

Every layout replays identical seeded parameter sequences on the retained
model. All measured likelihood values and final histograms are checked against
the same scenario's reference. With fewer than 842 local calls, the short local
scenarios sample only part of an 842-coordinate model; global calls still change
all variable coordinates. Original-space and eigen-space fit coordinates follow
the model's existing conversion rules.

The diagnostic API is
`BackendManager.setDiagnosticCpuCacheTraversal("Indexed" | "Auto")`. It changes
only the traversal policy between synchronous propagations, preserving caches
and model storage. Normal fits use Auto. `--benchmark weights` retains the
weight-factor comparison, with Auto cache traversal in every case.

## Timing and interpretation

Each measured call exports:

- `cacheSeconds`: complete response-cache stage duration.
- `cachePreparationSeconds`: candidate counting, parameter bookkeeping and any
  concatenated-index construction.
- `cacheExecutionSeconds`: worker dispatch, evaluation and join.
- `cacheCandidates`: responses potentially affected by raw parameter changes.
- `linearCacheTraversal`: whether the linear path was actually selected.

Candidate counts include responses skipped because their transformed input is
unchanged. Execution durations include scheduling and memory stalls, not just
arithmetic. Total cache time includes small boundaries outside the two subphase
timers. Sampled OS/profile exports are disabled in the benchmark. Constant calls
have no execution job, so their execution time is zero.

The summary reports median preparation/execution/cache durations and cache/full
LLH speedups against Indexed for the same scenario, thread count and round.
It also retains weight/histogram timings to expose downstream locality effects.
A global Auto case with no linear calls indicates that the model did not reach
the density threshold. Do not infer a gain from a subphase alone or from runs
on different nodes. No hardware bandwidth or cache-miss counters are collected.

## Validation

The dedicated C++ regression compares both paths bit for bit with all supported
cached spline/graph types, normalization, mirrors, extrapolation, response bounds,
caps, external weights, constants, unchanged inputs, sparse/dense transitions,
constant-dominated fallback and 1/2/5 threads. Existing smoothing, variable-response,
external-weight and NaN tests also run with the new default. The Python scenario
regression checks both modes, actual traversal selection, phase timings, replay,
likelihoods and histograms with original and eigen coordinates. The selected
109-test C++ suite passes (excluding the previously failing RootReaderCleanup
fixture), as do the profiling/scenario and Asimov Hesse regressions.

This change addresses traversal and index copying. It does not yet deduplicate
input transforms, compact descriptors, specialize dial-type batches or change
cache worker scheduling. A speedup on the cluster remains to be measured.

## Short local OA2024 observation

A Release check on the local OA2024 input used four threads and one retained
model: 416,034 events and 6,561,717 event/dial references. It compared both
policies in both orders, with two trials of five global calls per case, thirty
calls per local/constant trial and three warmup evaluations. All sixteen cases
passed likelihood and histogram checks (760 timed calls total). Results are
stored locally in `/tmp/gundam-weight-factors/cache-oa2024`.

| Global cache stage (ms) | Round 1 | Round 2 |
|---|---:|---:|
| Indexed preparation | 0.290 | 0.292 |
| Auto preparation | 0.0012 | 0.0011 |
| Indexed worker execution | 37.396 | 37.179 |
| Auto worker execution | 5.953 | 5.842 |
| Indexed complete cache stage | 37.703 | 37.473 |
| Auto complete cache stage | 5.954 | 5.843 |

The observed cache speedups are 6.33 and 6.41. The much larger improvement in
worker execution than in preparation supports prioritizing traversal/locality,
but does not distinguish bandwidth, cache misses and write sharing without
hardware counters. Local1/local2 calls remained on the targeted path in this
short prefix sample. Local/constant complete-evaluation medians stayed within
about 3% between policies. This short Mac result is not a cluster speedup
prediction, and short local sweeps do not cover all parameters.

## CCLyon confirmation, 21 October 2026

The paired cluster report is
`cpu-weight-benchmark-20261021-000207-2216521`, revision `gd40998ff`, on
`ccwslurm0319`. Eight threads and Dynamic x2 weights share one model/process
(PID 2216522), with 416,034 events, 3,529,068 cached dials, 6,561,717 event/dial
references and 842 variable fit coordinates. Initialization took 240.8 seconds.
This remains the smaller model, not the earlier 2.4-million-event input.

The sixteen cases contain 2,480 timed evaluations, including 80 global calls.
Independent checks of the raw files confirm matching sequence digests, candidate
counts and likelihood values across modes/rounds within the documented
likelihood tolerance. Final histogram numbers compare exactly. Global calls
have 3,528,913 candidates each; all forty Auto global calls select linear
traversal, and Indexed never does. Constant/local calls retain the targeted
path in this short sample. Local1/local2 cover only a prefix of the coordinate
list, so they do not establish performance for every possible sparse update.

The following are medians of the two trial medians, in milliseconds:

| Global stage | Indexed round 1 | Auto round 1 | Indexed round 2 | Auto round 2 |
|---|---:|---:|---:|---:|
| Cache preparation | 1.312 | 0.0036 | 1.338 | 0.0033 |
| Cache execution | 130.813 | 36.397 | 131.566 | 32.743 |
| Complete cache stage | 132.147 | 36.401 | 133.045 | 32.747 |
| Complete LLH evaluation | 140.981 | 44.661 | 142.891 | 40.465 |

Cache speedups are **3.63x and 4.06x**, corresponding to 72–75% lower latency.
Complete evaluation speedups are **3.16x and 3.53x**, or 68–72% lower latency.
The benefit persists in reversed order. The Auto round-1 cache trial medians
are 40.312 and 32.491 ms; both still substantially beat Indexed's 133.995 and
130.299 ms. This is a short diagnostic, not a whole-fit or MCMC throughput
measurement.

The gain is mainly in worker execution, not the approximately 1.3 ms saved on
candidate-list preparation. This supports retaining the new traversal but does
not distinguish cache misses, bandwidth, write sharing or worker balance
without hardware/worker-level measurements. The old approximately 160 ms
observation came from a different node/run; use this report's paired
132–133 ms baseline when attributing the improvement.

Local complete-evaluation medians vary by approximately -6% to +3% between
policies across these short cases, while their cache algorithms remain the same
and cache medians are approximately 0.04–0.06 ms. There is no reproducible local
regression in both rounds here. Constant evaluations have no worker cache job.
These observations should not be read as proof of universal no-regression.

The sum of all measured evaluation durations is **24.55 seconds**, including
7.47 seconds for global cases. This excludes initialization, warmup, priming,
parameter setters and output, but demonstrates that the shorter diagnostic
budget avoided the previous hour-long global workload.

Auto already remains the normal default; no backend change or additional
cluster run is required to adopt the improvement. Response-cache execution still
occupies approximately 81% of a global LLH evaluation. Sharing equivalent input
transformations is a next optimization candidate; the current report does not
measure its prospective benefit.
