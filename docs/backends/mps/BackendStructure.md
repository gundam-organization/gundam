# MPS backend structure

The MPS backend executes numerical propagation on an Apple GPU through Metal.
It prepares a packed device model from the common backend representation and
retains GPU buffers for repeated evaluations. Parameters and other changing
inputs are supplied at propagation time; model structure is prepared during
the build.

Metal kernels use single precision. Converting published results to host doubles
does not recover precision lost during device calculations. This differs from
the CPU backend's double-precision execution.

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
