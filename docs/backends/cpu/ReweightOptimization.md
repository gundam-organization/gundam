# CPU reweight optimization

The CPU backend separates updating unique dial responses from multiplying those
responses into event weights. Both stages use the resident immutable model and
retained output storage described in the [Backends library](../BackendsLibrary.md).

## Dial-response cache

Each eligible single-input dial has a cached response, its last transformed input
and a validity flag. Parameter-to-dial index lists identify affected responses
when raw parameter values change. A candidate is evaluated only if its cache is
invalid or its transformed input differs from the cached input. This preserves
input mirroring and avoids recalculation when different raw inputs map to the
same effective value. NaN inputs retain the evaluator's existing comparison rules.

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

## Scope of specialization

| Optimization | Current implementation |
|---|---|
| Dense response-cache traversal | Automatic sequential traversal, with sparse indexed fallback. |
| Shared model and compact weight data | Shared CPU model storage and separate hot event-weight descriptors. |
| Weight-loop specialization | Precompiled cached/mixed, capped/uncapped and scheduling variants. |
| Shared identical input transformations | Not implemented: transformations and cached input comparisons remain per dial. |
| Compact/type-specialized response updates | Not implemented: response updates still read generic dial descriptors and dispatch through the shared evaluator. |

Sharing equivalent input transformations could reduce repeated parameter loads,
mirroring and input comparisons, and reduce per-dial input bookkeeping. Equivalent
inputs must include the parameter identity and transformation settings. Any such
lookup must preserve the dense sequential dial traversal; revisiting all dials
in parameter-group order would give up its locality advantage.

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

`222CpuWeightSchedulingBenchmark.py --benchmark cache` compares `Indexed` and
`Auto` on the same loaded model with Dynamic x2 weights. It replays identical
constant, local and global parameter sequences, checks likelihood/histogram
agreement and exports cache preparation, worker execution and complete-evaluation
times. The global workload has a small independent call cap. Runtime diagnostic
switches are comparison tools, not fit tuning requirements; normal execution
uses Auto response traversal and Dynamic x2 weight scheduling.

TTree loading uses the effective limit from the global thread count and
`nbMaxThreadsForLoad`. A limit of one or two skips the raw I/O probe and uses
that many readers. With a higher limit, inputs below 500 MB use two readers;
larger inputs benchmark 2, 3, 4, 6, 8, ... readers, including enabled friend files.
Keeping at least two readers allows I/O to overlap with decompression and event
processing. See [dataset loading](../../configuration/DatasetLoader.md) for the
probe limits and selection policy.
