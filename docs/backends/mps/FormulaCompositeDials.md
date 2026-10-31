# MPS formula and composite dials

`Formula` and `Composite` dials run on the Metal GPU using the existing event
propagation executor. Configuration expressions, named inputs and graph layout
are identical to CPU. Both `cachedResponses` and `recomputeResponses` select
this executor when a formula or composite is present; neither uses a host
fallback for these dials.

`EngineLayout` compiles formulas once with `FormulaCompiler` and flattens nested
composites. `DevicePropagationModel` validates and packs the resulting programs,
input bindings and constants into fixed-width descriptors without host pointers.
`FormulaTypes.h` supplies the opcode definitions shared by C++ and Metal.
`MpsDialPrograms.metal` interprets those instructions directly on the device.

Formula inputs are resolved during packing to parameter indices, earlier graph
scratch slots or loaded event constants. Event inputs keep their snapshot
semantics, including mirroring already applied by `EventFormula`; they do not
read an observable after migration. Parameter/response mirrors remain on their
graph edges. Each node's response limits and smoothing are applied before its
result becomes available, including at nested-composite boundaries and at the
root. Graph, spline, normalization, shift, bilinear and bicubic leaves use the
same shared interpolation functions as ordinary MPS observable dials.

Each root formula/composite owns a disjoint range in a private GPU scratch
buffer. Formula stacks and graph intermediate values do not overlap. Scratch
is allocated at build time, sized from the validated program, and reused across
propagations. There is no fixed small device-stack limit or per-evaluation
allocation. Packing checks the 32-bit index limit and the executor checks Metal
buffer allocation limits. Program inputs, instructions and constants are uploaded
once; host packing storage is released when no longer needed.

The dial cache supports arbitrary input arity. Dials are grouped by sorted
parameter dependencies so a change to any input invalidates a shared response
exactly once. Event-only formulas and parameter-free composites are evaluated
on the initial pass and reused when caching is enabled. Observable invalidation,
bin migration, smoothing contributions and host readback use the existing MPS
pipeline.

## Numerical behavior

All program values use the MPS compensated `DoubleFloat` representation, with
approximately 48 significant bits and float32 exponent range. Arithmetic,
integer powers, comparisons, mirrors and response supervision retain the low
part. The function subset matches the CPU compiler: exponential, logarithmic,
square-root, trigonometric, inverse trigonometric, hyperbolic, error, rounding,
power and min/max operations.

`FormulaMath.h` implements compensated elementary functions using range
reduction, convergent series and corrections to native float estimates. Tests
compare ordinary finite arguments against CPU doubles. Results are not promised
to be bit-identical to CPU/ROOT. Overflow, underflow and subnormal behavior remain
limited by Metal float arithmetic. Exponential arguments outside [-87, 88],
trigonometric arguments with magnitude above 1e6, hyperbolic sine/cosine
arguments with magnitude above 80, and special-value estimates
use native float functions; these cases do not retain the same compensated
accuracy. Applications requiring double-range or large-argument transcendental
precision should use CPU. The Eb migration formula uses compensated arithmetic
without these transcendental fallbacks.

## Validation

`unitTest_MpsFormula.cpp` covers packing validation, compensated math, every
opcode, three-parameter sparse cache updates, repeated inputs, constant/event
inputs, and both execution modes. The composite interpolation regression also
compares all supported leaf types against the native engine on Metal.

`tests/fast-tests/234MpsFormulaCompositeDials.py` reuses the CPU/ROOT formula and
composite matrices with MPS in both modes and asserts that event propagation is
active. It covers event inputs, nested graphs, binning constants, input ordering,
mirroring, response smoothing, and formulas feeding splines or migrated
observables. CUDA continues to reject these dial types explicitly.
