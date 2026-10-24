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
`propagationMode` is the only MPS backend configuration option. Within-evaluation
sharing is automatic: the packed executor evaluates spline/graph responses used
at least eight times once per propagation and shares them between events. This
does not retain responses between propagations in `recomputeResponses` mode.

## Execution and storage

The MPS backend executes numerical propagation on an Apple GPU through Metal.
It prepares a packed device model from the common backend representation and
retains GPU buffers for repeated evaluations. Parameters and other changing
inputs are supplied at propagation time; model structure is prepared during
the build.
Construction-only host copies of uploaded payloads, variables and binning tables
are released after the uploads complete. Host metadata needed for cache lookup,
event-state materialization and diagnostics remains available.

The resident `eventPropagation` executor uses compensated pairs of floats
(about 48 significant bits) for parameters, dial responses, event-weight
products, contribution fractions and histogram sums / squared sums. Both
reduction stages preserve the pairs, and host readback reconstructs doubles
from both components. This avoids float32 rounding noise in the finite
differences used by Migrad and Hesse. It doubles the corresponding numerical
buffers relative to float32; it is not native double-precision GPU arithmetic.

The scalar fixed-bin `packedWeights` executor selected by `recomputeResponses`
still uses float32 weights and histogram outputs. Its numerical precision is
unchanged. The minimizer receives a recommended relative function precision of
`1e-12` for the resident executor and float epsilon (`1.1920928955078125e-7`)
for the packed executor. These are numerical differentiation defaults, not
universal error bounds; an explicit `minimizerConfig.functionPrecision` takes
precedence. See [toy precision validation](../../guides/MpsToyPrecision.md).

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
Only the compact descriptors are uploaded to the GPU.

Spline payloads are placed in first-use order across logical groups of 32 event
threads. Within each group, families and dial ordinals are visited in kernel
order before advancing across the event lanes. A payload is copied only once;
its descriptor receives the new offset. Payload values, knot ordering and spline
evaluation are unchanged. Shared dials retain one payload and the same response
slot. This layout requires no extra resident GPU storage in normal runs.

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

The bin search also rejects a candidate as soon as its remaining intervals cannot
change the result. Strict binning only evaluates hard containment and stops at
the first miss; it does not calculate unused smoothing fractions. For smoothed
binning, early rejection requires both failed hard containment and a zero
accumulated fraction. A point outside the hard bin may still contribute through
the smoothing support, and a hard match must still be recorded even when its
fraction rounds to zero. The order of nonzero fractional products and bin
priority are preserved.

The resident event-propagation path caches dial responses by their immutable
parameter/external-block dependencies. Build-time lists group original response
indices by parameter or external block. Each call selects changed groups from
the host parameter snapshot and published external generations. Sparse updates
upload a few range descriptors and dispatch only the affected response indices;
updates affecting more than half the dials keep the direct full scan. Constant
Shift responses are initialized by the first full pass. Rebuilds and failed
propagations invalidate the whole cache. `propagationMode: recomputeResponses`
disables this inter-evaluation response/weight cache.

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
The final per-bin reduction also preserves compensated pairs. Other SIMD widths
use the serial compensated kernel. The arithmetic is validated against CPU and
reference histograms; binary identity is not required.

These optimizations are part of the production implementation.

`getDiagnosticMpsEventPropagationState()` returns the last evaluated-dial count,
event-weight update flag and index-rebuild flag without scanning the model.
`setDiagnosticMpsStageProfiling(True)` enables separate synchronized GPU stage
probes; `False` restores ordinary execution. The profiling benchmarks retain
ordinary timing passes and separate probes for the production implementation.
