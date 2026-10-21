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
smoothing-table index. Event references, spline payloads, response slots, and
multiplication order retain their existing numbering and semantics.

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
