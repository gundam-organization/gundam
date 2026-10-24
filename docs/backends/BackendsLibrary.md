# Backends library

## From configuration to computation

The original engine connects the user's configuration to an in-memory model:
parameters, events, dials, samples and their relationships. Its object structure
expresses the requested physics and keeps the full range of features available.
It remains responsible for constructing and connecting these objects, whichever
execution backend is selected.

That organization also serves the original propagation path, but is not always
the most efficient representation for repeated numerical evaluation. Following
relationships between objects introduces indirection, and generic execution must
accommodate different dial types and optional behavior. These costs matter when
a fit repeats the same operations over millions of events and dial references.

Backends provide hardware-specific execution of this model. They arrange data
and operations for the target processor: compact arrays, reusable numerical
caches, specialized loops and an appropriate division of parallel work. The
goal is to prepare the execution structure once, then reuse it as parameter
values change throughout the fit.

The separation lets configuration and physics remain in the engine while each
backend chooses a memory layout and execution strategy suited to its hardware.
Shared numerical semantics keep these implementations consistent; a common
model does not require identical storage, scheduling or floating-point precision
on every device.

## Available backends

Implementation details, numerical precision and performance choices are documented
with each backend:

- [CPU structure](cpu/BackendStructure.md) and [optimization choices](cpu/ReweightOptimization.md).
- [MPS structure](mps/BackendStructure.md).
- [Multi-input dials](MultiInputDials.md).

## Compatibility

These tables describe the `CPU` and `MPS` backends selected through
`backendManagerConfig.type`. The original propagation path and the legacy
`CacheManager` (including its CUDA implementation) are separate execution paths;
their capabilities must not be inferred from these tables.

Legend: ✅ supported; ❌ unsupported; ⚠️ supported with the stated restriction.

### Dial types

Compatibility depends on the concrete dial produced by the loader. Both backends
use the same `EngineLayout` conversion, which rejects unsupported dial classes
with `Unsupported dial type for EngineView`. MPS additionally rejects
`Tabulated` and `Kriged` during backend construction. There is no automatic fallback to
the original engine for these classes.

| Configuration or concrete dial | CPU | MPS |
|---|---|---|
| `Normalization` (`Norm`) | ✅ | ✅ |
| Constant response (`Shift`) | ✅ | ✅ |
| `Spline`: `CompactSpline` | ✅ | ✅ |
| `Spline`: `UniformSpline` | ✅ | ✅ |
| `Spline`: `MonotonicSpline` | ✅ | ✅ |
| `Spline`: `GeneralSpline` | ✅ | ✅ |
| `Graph` | ✅ | ✅ |
| `Surface` / `Bilinear` | ✅ | ✅ |
| `Surface` / `Bicubic` | ✅ | ✅ |
| `ExternalWeight` (`ExternalWeightDispatcher`) | ✅ | ✅ |
| ROOT objects (`RootFormula`, `RootSpline`, `RootGraph`) | ❌ Heavy C++ objects, unsuitable for the flat backend model | ❌ Strictly incompatible with Metal device execution as C++ ROOT objects |
| `Tabulated` | ⚠️ Table updated on host at each propagation | ❌ |
| `Kriged` | ⚠️ Table updated on host at each propagation | ❌ |
| `CompiledLibDial` | ❌ | ❌ |
| `Polynomial` | ❌ | ❌ |

ROOT input objects (`TGraph`, `TSpline3`, `TH2`) and supported TArray branch pairs
are loading formats, not backend dial types. Their compatibility follows the
concrete dial constructed from them. Spline slope options are also resolved
during construction, before backend execution.

The CPU backend reads shared `Tabulated` and `Kriged` tables after the host
factories update them during propagation preparation. Fixed indices and
coefficients are flattened once; table values are supplied as dynamic blocks.
Responses are evaluated on each propagation, including when parameter values
are unchanged. Each table publication advances its generation so observable
and binning caches also see the update.

### Engine features

Here, ✅ means the backend can handle the feature; it does not imply that
every stage runs on the GPU. MPS support applies to both `cachedResponses` and
`recomputeResponses` unless indicated otherwise.

| Feature | CPU | MPS |
|---|---|---|
| Event weights and histogram sums / squared sums | ✅ | ✅ |
| Statistical likelihood | ⚠️ on host | ⚠️ on host |
| Separate `SampleLikelihoods` output | ❌ Only the total is stored | ❌ Only the total is stored |
| Parameter inputs and per-input mirroring | ✅ | ✅ |
| Dial extrapolation | ✅ | ✅ |
| Hard response limits and response smoothing | ✅ | ✅ |
| Event reweight cap | ✅ | ✅ |
| Observable responses (`applyToVariable`) | ✅ | ✅ |
| Dynamic binning / event migration | ✅ | ✅ |
| Smoothed binning / fractional contributions | ✅ | ✅ |
| External-weight updates | ⚠️ Produced on host | ⚠️ Produced on host, copied to GPU buffers |
| Event variables as ordinary dial inputs (`DialInputSource::EventVariable`) | ❌ | ❌ |
| Numerical precision | Double (64-bit) | ⚠️ Float32 outputs; ~48 significant bits for observables and resident histogram accumulation (float pairs) |
| Hardware requirements | ✅ Host CPU | ⚠️ Apple GPU with Metal |

MPS chooses its executor automatically: surfaces, observable/binning propagation
and reweight caps use `eventPropagation`; compatible scalar fixed-bin models can
use `packedWeights` in `recomputeResponses` mode. These internal choices do not
add support for dial classes rejected by `EngineLayout`. See
[MPS response propagation modes](mps/BackendStructure.md#response-propagation-mode).

The compatibility rules are implemented in
[`EngineLayout.cpp`](../../src/Backends/Common/src/EngineLayout.cpp),
[`CpuBackend.cpp`](../../src/Backends/Implementations/CPU/src/CpuBackend.cpp) and
[`MpsBackend.mm`](../../src/Backends/Implementations/MPS/src/MpsBackend.mm).

## Connection to the engine

`BackendManager` connects the likelihood interface to a backend: it prepares
runtime inputs, requests propagation and materializes the outputs needed by the
fitter or host-side consumers.

## Model, inputs and outputs

`EngineLayout` separates a flat numerical model (`EngineView`) from bindings to
the original GUNDAM objects. The model contains dial descriptors and payloads,
event-to-dial indices, event-weight descriptors, observable/binning metadata and
likelihood definitions. It is built once for a fixed model layout.

At each propagation, the manager supplies parameter values, external-weight
buffer views and generations, and observable revisions or host values when
needed. Numerical executors consume these inputs without traversing the original
event and dial object graph. Changes to model structure require rebuilding the
layout; changes to parameter values do not.

Each backend decides how to retain, pack or upload the model for its hardware.
Mutable caches and outputs belong to the executor; the common representation
keeps numerical definitions separate from those execution choices.

A propagation token identifies its outputs. Readiness and materialization requests
control which event weights, histograms, observable state and likelihood results
are available to the caller, including any necessary device-to-host transfer.

## Cache responsibilities

| Data | Purpose and lifetime |
|---|---|
| Event/dial associations | Record which unique dials affect each event, preserving their product order; retained with the model. |
| Dial responses | Reuse eligible numerical responses until their effective inputs change; managed by the executor. |
| Observable/binning state | Track relevant parameter, external-buffer and observable revisions so dependent state is updated when needed. |
| Numerical outputs | Publish results for a propagation token; retained storage can be overwritten by later evaluations. |

Cache invalidation follows the dependencies of the stored quantity. Reusing a
dial response does not imply that downstream event weights or histograms can be
reused. External-weight generations and observable revisions identify changes
that parameter values alone cannot describe; producers must report them through
the input interfaces. Each backend documents which intermediate results it caches.

Shared semantics define dial evaluation, input transformations, response bounds
and smoothing. Executors may specialize traversal, scheduling and memory layout
while preserving these numerical rules within their supported precision and
capabilities.
