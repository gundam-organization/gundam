# CPU reweight optimization

The CPU backend separates updating unique dial responses from multiplying those
responses into event weights. Both stages use the resident immutable model and
retained output storage described in the [Backends library](../BackendsLibrary.md).

## Dial-response cache

Each eligible single-input dial retains its own response and validity flag.
Parameter-to-dial index lists identify affected responses when raw parameters
change. A candidate is evaluated only if its response is invalid or its effective
input differs from its previous value. This preserves mirroring and avoids
recalculation when different raw inputs map to the same effective value. NaN
inputs retain the evaluator's existing comparison rules.

### Shared effective inputs

At construction, the CPU groups cacheable inputs by parameter identity, mirror
activation, and mirror minimum/range. Active mirror settings use bitwise keys;
unused bounds do not distinguish unmirrored inputs. The temporary lookup table
is discarded after construction. Dials retain their original order and reference
a group through a 32-bit index; their spline data and responses stay separate.

When a parameter changes, each affected group computes its transformed input
and equality test once. Workers read that value and change flag while retaining
the response's own validity check. Unchanged groups avoid per-dial transforms;
NaN effective inputs still compare unequal and are reevaluated when their raw
parameter changes. Group state is published only after the worker job joins,
so workers never update shared group values concurrently.

The group-index array and per-group state **replace** the per-dial double input
cache. Construction checks actual retained capacities, including group-to-parameter
indices; if they would cost at least as much as the old input cache, it falls
back to per-dial evaluation/storage. Models without enough reuse therefore do
not retain both representations. An early distinct-group bound also limits
construction work for models with unique transformation settings.

This shares input preparation without traversing dials by parameter group,
preserving the sequential response-cache path's locality.

Immutable `Shift` responses, including response limits and smoothing, are computed
at model construction. They retain their position in the event product rather
than being folded into the base weight. External weights and other dials outside
the response-cache eligibility rules use the mixed evaluation path.

### Sparse and dense updates

The default `Auto` policy first counts candidates from the parameter-to-dial
lists and records changed parameters in a small byte mask:

- With no candidates, it skips the response-cache worker job.
- If fewer than half of all dial descriptors are candidates, workers traverse
  the concatenated indices of affected dials, in parameter-list order.
- Otherwise, workers traverse contiguous ranges of dial descriptors in increasing
  index order, skipping immutable shifts, uncached dials and unchanged parameters.

The dense path avoids constructing the candidate-index array and improves the
locality of descriptor and response accesses. A change to one parameter can be
dense if it affects many dials. The sparse path avoids a full-model scan for
small changes or models dominated by constants/non-cacheable dials. The selection
is automatic; no numeric fit configuration is required.

Both paths use the same evaluator and write the same response slots. They retain
transformed-input equality checks, mirroring, extrapolation, bounds, smoothing
and double precision. The dense path adds no duplicate descriptor, payload or
response buffer. Its parameter mask uses one byte per physical input parameter.

## Event weights and scheduling

Weight descriptors are separate from observable/binning metadata and contain
only the base weight and the event's dial-index range (24 bytes on a 64-bit
build). Every event's response product is computed in its original dial order;
any reweight cap applies before multiplication by the base weight.

At construction, the backend selects precompiled kernels for fully cached or
mixed responses, with or without a cap, and dynamic or fixed scheduling. These
choices remove corresponding branches from the inner loops. There is no JIT
compiler involved. Mixed kernels retain access to non-cached response evaluators.

The default scheduler prepares **two contiguous blocks per worker**, balanced
using estimated work `1 + dialCount`. Workers claim blocks through a shared
atomic counter and write disjoint output slots. Single-thread and very small
workloads use fixed ranges. Only range metadata is replicated; event, response
and payload buffers are shared. Histogram filling and reduction keep their own
execution order.

An experimental diagnostic API also selects kernels with two or four interleaved
cached events; normal fits retain the scalar kernel. These independent
accumulators preserve product order within each event and reuse the same buffers.
Mixed responses retain scalar execution. The dedicated
[interleaving benchmark](../../../tests/benchmarks/README.md#experimental-interleaved-event-weights)
checks weights and histogram bits and compares against scalar on the same loaded
model. This is an experiment, not an enabled optimization or a JIT compiler.

## Scope of specialization

| Optimization | Current implementation |
|---|---|
| Dense response-cache traversal | Automatic sequential traversal, with sparse indexed fallback. |
| Shared model and compact weight data | Shared CPU model storage and separate hot event-weight descriptors. |
| Weight-loop specialization | Precompiled cached/mixed, capped/uncapped and scheduling variants. |
| Shared identical input transformations | Prepared equivalent-input groups, with one transformation/comparison per changed group and a memory-bounded per-dial fallback. |
| Compact/type-specialized response updates | Not implemented: response updates still read generic dial descriptors and dispatch through the shared evaluator. |

Shared inputs remove repeated parameter/input loads, mirroring and equality
checks in grouped execution. Response-specific bounds, smoothing, spline
interval selection and numerical evaluation remain per dial. Generic immutable
input descriptors are still part of the common model; this optimization does
not compact or duplicate that representation.

A smaller hot response descriptor and batches specialized by dial type or
response policy could reduce metadata traffic and repeated dispatch. Those are
separate changes from the existing compact event-weight descriptors. They must
retain uncommon dial features and avoid adding another full-size backend copy.
Their benefit depends on the model and requires a paired measurement; neither
improvement is implied by the existing traversal optimization.

## Verification and diagnostics

CPU regression tests compare weights, histogram sums and squared sums bit for
bit across the indexed and automatic response traversals, including sparse/dense
transitions and mixed features. The numerical contract also preserves event
product order and histogram reduction order across weight schedulers.

Performance scripts live in `tests/benchmarks`; see the
[benchmark instructions](../../../tests/benchmarks/README.md). The resident
weight-scaling benchmark loads the model once and varies only active weight
workers. It separates ordinary stage timings from per-worker wall-time probes,
retains the histogram pool and checks likelihood/histogram agreement. These
worker controls are diagnostic API calls; the default fit policy is unchanged.

`222CpuWeightSchedulingBenchmark.py --benchmark cache` compares `Indexed` and
`Auto` on the same loaded model with Dynamic x2 weights. It replays identical
constant, local and global parameter sequences, checks likelihood/histogram
agreement and exports cache preparation, worker execution and complete-evaluation
times. The global workload has a small independent call cap. Runtime diagnostic
switches are comparison tools, not fit tuning requirements; normal execution
uses Auto response traversal and Dynamic x2 weight scheduling.

`--benchmark inputs` compares `PerDial` and `Shared` input evaluation, retaining
Auto traversal and Dynamic x2 weights. It loads the model once, replays the same
sequences and retains the short global-call cap. For example:

```sh
python3 /path/to/gundam/tests/benchmarks/222CpuWeightSchedulingBenchmark.py \
  --benchmark inputs --config configOA2024.yaml --threads 8 \
  --scenarios local1 --calls 1 --warmup 3 --trials 1 --rounds 1 \
  --override override/v13Prod.yaml override/backendCpuTest.yaml
```

Timed evaluations use the same output policy as Migrad/HESSE: automatic
copies into host event/histogram objects are disabled. The likelihood is still
updated every call; the final histogram is explicitly published outside timing
for comparison. Results record `autoMaterializeDuringTiming: false`. Older
reports without that field included automatic host copies, so their complete
evaluation times should not be compared directly with the fit loop. Cache and
weight stage timings do not include those copies. The input override selects
the dataset; the script adds no event-count limit.

Local workloads automatically cover every variable fit coordinate: `local1`
uses at least N calls per trial and `local2` at least ceil(N/2). `--calls` can
request additional calls; it cannot truncate this coverage. Coordinates are
visited in manifest order, with reproducible bounded random displacements
around the initialized point. Previously moved coordinates retain their small
displacement until the trial ends. Eigen coordinates may change several physical
parameters, so these are controlled workloads rather than actual Migrad steps.

With 842 coordinates, this short example runs 1,684 timed evaluations, plus
warmup and priming, using one model load and no global updates. Each `local1`
case writes a `*-coordinate-costs.json` file with coordinate names, original/eigen
space, sample counts, stage timings and affected-dial counts. Raw results record
the coordinate indices per call; the summary records coverage. Use more trials
for stable per-coordinate measurements. Global calls remain independently capped
by `--global-calls` when that scenario is selected.

`cacheInputGroups`, per-call `inputTransforms` and `sharedCacheInputs`
identify whether sharing was available and used. Group metadata/state remain
common to both modes: PerDial repeats transformations/comparisons in workers,
while Shared reads prepared group values. The comparison isolates input reuse
with the same storage, not the entire previous cache representation. The
transform counter includes group preparation in both modes and the additional
per-dial transforms in PerDial mode. Group publication is included in cache
execution time. Neither mode adds a second full-size input cache.

If grouping falls back, the modes intentionally use the same per-dial path;
`cacheInputGroups` is zero and no sharing gain should be inferred. Runtime
switches preserve initialized response state. Regression coverage includes all
cached dial types, different mirror settings, unchanged effective inputs, NaNs,
mode switches, unique-input fallback, and original/eigen fit coordinates.

TTree loading uses the effective limit from the global thread count and
`nbMaxThreadsForLoad`. A limit of one or two skips the raw I/O probe and uses
that many readers. With a higher limit, inputs below 500 MB use two readers;
larger inputs benchmark 2, 3, 4, 6, 8, ... readers, including enabled friend files.
Keeping at least two readers allows I/O to overlap with decompression and event
processing. See [dataset loading](../../configuration/DatasetLoader.md) for the
probe limits and selection policy.
