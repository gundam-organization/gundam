# MPS backend structure

Supported dial types and engine features are listed in the
[compatibility tables](../BackendsLibrary.md#compatibility).

## Response propagation mode

Select the response strategy in the MPS backend configuration:

```yaml
fitterEngineConfig:
  likelihoodInterfaceConfig:
    backendManagerConfig:
      isEnabled: true
      type: MPS
      backendConfig:
        propagationMode: cachedResponses
```

`cachedResponses` is the default, including for scalar dials and fixed bins.
It retains dial responses between evaluations and updates only responses whose
parameter inputs or external-weight generations changed. It can also skip event
weight updates when their dependencies are unchanged. This usually favors local
parameter changes. Existing scalar configurations that omitted this option now
use this strategy instead of the previous packed weight executor.

`recomputeResponses` recalculates responses and event weights on every evaluation.
Responses shared by events may still be evaluated once within an evaluation.
This strategy can be faster when many parameters change together; benchmark the
actual model before selecting it. Neither mode depends on the minimizer.

Both modes support scalar and two-input surface dials, observable propagation,
dynamic binning, external weights and event reweight caps. These are response
strategies, not switches that enable or disable event migration. The CPU backend
does not expose this option; it already caches responses and adapts its cache
traversal to the number of changed dependencies.

The monitor displays `MPS/cachedResponses` or `MPS/recomputeResponses`. Startup
logs and `getDiagnosticMps()` also identify the internal executor. Cached mode
uses `eventPropagation`. Recompute mode uses `packedWeights` for compatible
scalar fixed-bin models, and `eventPropagation` with its inter-evaluation
response cache disabled for surfaces, migration and reweight caps. Thus a model
requiring the general executor still honors the requested recomputation strategy.
The older `enableCachedDialResponses` setting remains available: `false`
disables response caching in cached mode, or within-evaluation sharing in the
packed executor. Prefer `propagationMode` to select the inter-evaluation strategy.

## Execution and storage

The MPS backend executes numerical propagation on an Apple GPU through Metal.
It prepares a packed device model from the common backend representation and
retains GPU buffers for repeated evaluations. Parameters and other changing
inputs are supplied at propagation time; model structure is prepared during
the build.

Weights, contribution fractions and published histogram values use single
precision. Event observables and interval arithmetic use compensated pairs of
floats. Parallel event-propagation histogram reductions also accumulate in
compensated pairs before publishing float partial sums. Converting published
results to host doubles does not recover rounding at these float boundaries.
This differs from the CPU backend's double-precision execution.

The backend interface separates execution from host materialization, so callers
can request the outputs they need. The shared model and manager responsibilities
are described in [Backends library](../BackendsLibrary.md).

The packed weight path uses 24-byte spline/graph descriptors and 16-byte
normalization/external-weight occurrences. These contain indices and response
thresholds. Optional smoothing coefficients live in a separate, deduplicated
table and are read only when a response crosses a transition threshold. A zero
smoothing index selects hard bounds. The spline descriptor retains extrapolation
and response-cache flags in its low two bits; its remaining bits hold the
smoothing-table index. Event references, descriptor indices, response slots, and multiplication order
retain their existing numbering and semantics.

The common 48-byte spline descriptors and 36-byte normalization/external
occurrences remain the backend-independent packing representation. MPS converts
them during model construction, then releases these temporary host arrays.
Legacy GPU descriptor buffers are allocated only when
`enableDiagnosticDescriptorComparison` is enabled. This diagnostic option allows
repeated A/B trials without reloading inputs; it is disabled by default.

`tests/benchmarks/225MpsBackendProfiling.py --compare-descriptors` enables that
option and alternates the legacy and compact layouts on one resident model.
The shared event loop is instantiated for both wire formats; interpolation and
response-smoothing arithmetic are shared as well. Each measured likelihood and
all final histogram bins must match exactly. Separate GPU stage probes run after
the ordinary trials. See [benchmark instructions](../../../tests/benchmarks/README.md).

Spline payloads are placed in first-use order across logical groups of 32 event
threads. Within each group, families and dial ordinals are visited in kernel
order before advancing across the event lanes. A payload is copied only once;
its descriptor receives the new offset. Payload values, knot ordering and spline
evaluation are unchanged. Shared dials retain one payload and the same response
slot. This layout requires no extra resident GPU storage in normal runs.

`tests/benchmarks/225MpsBackendProfiling.py --compare-payloads` retains both
payload layouts and their compact descriptors for exact A/B measurements. It
sets `enableDiagnosticPayloadComparison`, which is disabled in normal runs.
The benchmark switches both descriptor offsets and the payload buffer together,
including cached-response kernels. It alternates trial order and validates
every likelihood and all final histogram bins exactly.

Two additional arrangements were measured on OA2024 and discarded: interleaving
dial references across 32-event tiles, and sorting events by dial-count profiles
inside 256-event windows. They slowed evaluation on this dataset. Event order
and the per-event reference layout therefore remain unchanged.

Dynamic strict binning first checks the event's previous bin. At model build,
`BinningIndex::buildReusableBinPositions` certifies bins that cannot overlap an
earlier selection. CPU and MPS share this certification, including point and
category selections. MPS uploads a small lookup from global histogram bin to
packed bin descriptor; missing or unsafe bins have no reusable entry. The
lookup preserves sample offsets and bin identifiers independently of descriptor
positions.

The first propagation initializes the stored bins through the tree. Subsequent
finite observable updates test a certified previous bin with the same compensated
interval comparisons as the tree. If it no longer contains the event, or cannot
be certified, the existing search preserves the earliest matching bin. Returning
from outside the histogram also uses the tree. The shortcut still publishes the
sample revision and reconstructs contribution indices as before. It is disabled
for smoothed binning, whose fractions and neighboring contributions can change
while the hard bin remains the same.

`setDiagnosticMpsPreviousBinReuse(False)` selects the original tree search on an
initialized model; `True` restores the production default. The production
binning benchmark's `--compare-previous-bin` alternates the two modes on one
resident model, compares likelihoods and final histograms, then records separate
GPU stage probes. With smoothing enabled this flag leaves bin selection on the
existing path.

The bin search also rejects a candidate as soon as its remaining intervals cannot
change the result. Strict binning only evaluates hard containment and stops at
the first miss; it does not calculate unused smoothing fractions. For smoothed
binning, early rejection requires both failed hard containment and a zero
accumulated fraction. A point outside the hard bin may still contribute through
the smoothing support, and a hard match must still be recorded even when its
fraction rounds to zero. The order of nonzero fractional products and bin
priority are preserved.

`setDiagnosticMpsBinningIntervalPruning(False)` restores the full interval loop
for A/B measurements. `True` is the production default. The production benchmark
accepts `--compare-binning-intervals` to compare both loops on the same resident
model, independently of the previous-bin shortcut. The two comparison flags are
mutually exclusive.


The resident event-propagation path caches dial responses by their immutable
parameter/external-block dependencies. Build-time lists group original response
indices by parameter or external block. Each call selects changed groups from
the host parameter snapshot and published external generations. Sparse updates
upload a few range descriptors and dispatch only the affected response indices;
updates affecting more than half the dials keep the direct full scan. Constant
Shift responses are initialized by the first full pass. Rebuilds and failed
propagations invalidate the whole cache. `enableCachedDialResponses: false`
also disables this event-propagation response/weight cache.

Weight dependencies are tracked separately from observable dependencies. If only
observable inputs change, the event kernel updates variables and dirty flags
while retaining resident event weights. When neither needs an update, the event
kernel can be omitted. External blocks used by weights invalidate weights on a
new published generation. This cache belongs to the event-propagation path; the
packed weight-only executor keeps its existing implementation.

Rebinning compares each new contribution destination against the old destination
in that event's stable contribution slots, and compares the contribution count.
An unchanged destination list can retain the histogram index even if fractions
or the hard bin change. New/removed destinations, reordered slots, first
propagation and recovery after failure require rebuilding. Outside-support
contributions participate in this comparison. Variables, fractions and sample
revisions are still published independently of index reuse. Any changed event
list currently triggers a global rebuild; this is not a per-sample index.

Rebinning returns event-local counters and flags before the kernel's SIMD
collectives. This reconverges lanes after event-local early exits. Counts and the
rebuild flag are combined per SIMD group, then published with fewer global
atomics. Sample flags are combined for lanes sharing the first lane's sample;
other samples retain explicit writes, including groups crossing sample boundaries.

Histogram chunks contain up to 256 contributions. On pipelines with 32-wide SIMD
groups, a parallel kernel assigns one group to a chunk. Each lane accumulates
strided contributions, then a shuffle reduction combines compensated float pairs.
The existing final per-bin reduction remains in place. Other SIMD widths use the
serial compensated kernel. The parallel arithmetic is validated against CPU and
reference histograms with float tolerances; binary identity is not required.

`setDiagnosticMpsEventPropagationOptions(cacheDials, reuseIndex,
parallelHistograms, aggregateCounters)` controls the four features for resident
comparisons. All are enabled by default (subject to the cache configuration and
SIMD-width fallback). `getDiagnosticMpsEventPropagationState()` returns the last
evaluated-dial count, event-weight update flag and index-rebuild flag without
scanning the model. `226ProductionBinningProfiling.py --compare-resident-stages`
compares baseline, cached, indexed, parallel and aggregated variants, rotating
trial order on a single loaded model. Separate stage probes report the actual
kernels dispatched; skipped stages are absent.
