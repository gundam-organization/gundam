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
- [CUDA implementation and validation](../guides/CudaBackend.md).
- [Multi-input dials](MultiInputDials.md).
- [Small-fit and Hesse covariance regression](BackendFitCovariance.md).
- [CUDA production profiling](../../tests/benchmarks/README.md#cuda-production-profiling).
- [CUDA OA2024 profiling results and optimization priorities](CudaOa2024Benchmark.md).

## Compatibility

These tables describe the `CPU`, `MPS` and `CUDA` backends selected through
`backendManagerConfig.type`. The original propagation path and the legacy
`CacheManager` (including its CUDA implementation) are separate execution paths;
their capabilities must not be inferred from these tables.

Status as of 2026-10-24. Legend: ✅ implemented/supported; ❌ unsupported;
⚠️ implemented with the stated restriction or validation pending;
🚧 not implemented yet; — not applicable. Implementation status does not by
itself establish numerical validation on every model or device.

### Implementation progress

| Stage | CPU | MPS | CUDA |
|---|---|---|---|
| Factory selection and backend configuration | ✅ `CPU` / `cpu` | ✅ `MPS` / `mps`, `propagationMode` | ✅ `CUDA` / `cuda`, `deviceIndex`, `propagationMode` |
| Execution resources | ✅ Host worker pool | ✅ Metal device, buffers and command execution | ✅ Device, stream, resident buffers and timing events exercised on CC V100 |
| Model preparation | ✅ Flat host model and numerical caches | ✅ Packed model and GPU uploads | ✅ Native-double descriptors, payloads and fixed-bin index uploaded once |
| Propagation and result tokens | ✅ | ✅ | ✅ Synchronous CUDA executor and token invalidation |
| Host materialization | ✅ Event state, weights and histograms | ✅ Event state, weight and histogram readback | ⚠️ Lazy weight/histogram readback; no event-state output |
| Dial-response and event-state caches | ✅ | ✅ Depends on propagation mode | ⚠️ Sparse parameter-indexed response cache exercised on CC OA2024, matching the prior CPU reference; no event-state cache |
| Resident backend replacement | ✅ Unchanged fixed-bin model | ⚠️ Fixed-bin API available; MPS replacement not validated here | ✅ Both response modes and index layouts exercised with CPU in one OA2024 process on CC V100 |
| Numerical regression tests | ✅ CPU comparisons | ✅ MPS comparisons | ✅ Both modes compared against CPU on CC V100; error/retry tests passed |
| Small fit and full Hesse covariance regression | ✅ Original/CPU comparison and analytic covariance reference pass locally | — Not included in this regression | ✅ Both modes pass at CC, including the analytic covariance reference |
| Standalone CUDA infrastructure test | — | — | ✅ FP64 kernel validated on CC-IN2P3 V100, interactive and Slurm batch |

CUDA implements scalar and two-input Surface dials for event weights with fixed bins.
Its defaults are `propagationMode: cachedResponses` and
`weightLayout: warpInterleaved`. Index padding above 2x retains `eventMajor`.
`cachedResponses` (default) retains dial responses until any transformed input
changes; `recomputeResponses` evaluates every unique dial on every propagation.
Both modes recompute event weights and histograms, using native doubles. External
weights, tables and event-state propagation are rejected during
model preparation. No CPU propagation fallback is performed.

Bilinear/Bicubic surfaces use the shared interpolation and retain ordered x/y
inputs, per-input mirroring, extrapolation, response bounds and smoothing.
Disjoint dependency-pair groups prevent duplicate response writes when both
parameters change. Scalar-only models retain the specialized scalar kernels;
mixed models allocate the additional second-input cache only for surfaces.
Host packing/validation tests pass locally. The new `cuda-backend-surfaces`
device suite compares against native dials and CPU; CUDA compilation/execution
of this extension remains to be validated at the CC.

The host-side configuration/model tests pass locally without a toolkit. The
full CUDA test executable passed all five tests on CC-IN2P3 V100 in Slurm job
`62374030` on 2026-10-24: one device-lifecycle test and four propagation tests
(184 ms for the whole suite, not a performance benchmark). The scalar comparison
test checks both modes, with and without extrapolation/caps, at eight parameter
points per combination, using 521 events and all seven scalar dial types.
Weights, histogram sums/squared sums and host likelihoods agree with CPU within
the tests' tolerances: `2e-12 * max(1, abs(reference))` for weights/histograms
and `2e-11 * max(1, abs(reference))` for likelihoods. See
[`unitTest_CudaBackendPropagation.cpp`](../../tests/GTests/unitTest_CudaBackendPropagation.cpp).

Empty models, undialed events, lazy readback, obsolete tokens, numerical overflow
and recovery after errors also pass. This validates the synthetic test models,
not a complete fit or production-scale model. Only one GPU was visible;
switching between distinct devices and H100 execution remain unvalidated.
The standalone FP64 `cuda-smoke` test previously passed in interactive and batch
use as well.

The optional [fit/covariance regular tests](BackendFitCovariance.md) generate
540 weighted events in eight bins and fit three correlated parameters using
Migrad and Hesse. They compare the original engine with CPU, then CPU with
each CUDA mode, including full covariance/correlation matrices and eigenvalues.
An independent analytic covariance anchors all fits. The original/CPU comparison
passes locally in less than a second. The three reports returned from the CC also
pass: CUDA in both modes gives a maximum covariance difference normalized by
posterior standard-deviation products of `1.93e-10` and a maximum absolute
correlation difference of `1.14e-10`. All fits have status 0, covariance status 3,
and take 92 calls. This validates the small regression model, not the full
production OA2024 covariance.
Unavailable backends or devices produce skips by default, while
`GUNDAM_REQUIRE_CUDA_TESTS=1` requires GPU execution in a validation job.

### Dial types

Compatibility depends on the concrete dial produced by the loader. All three backends
use the same `EngineLayout` conversion, which rejects unsupported dial classes
with `Unsupported dial type for EngineView`. MPS additionally rejects
`Tabulated` and `Kriged` during backend construction. There is no automatic fallback to
the original engine for these classes. CUDA's scalar evaluator reuses common
numerical semantics; its scalar dial types passed the V100 comparison tests in
both propagation modes. 🚧 marks missing CUDA implementations, rejected at build time even when
the common layout can represent the dial. ❌ marks unsupported dial classes.

| Configuration or concrete dial | CPU | MPS | CUDA |
|---|---|---|---|
| `Normalization` (`Norm`) | ✅ | ✅ | ✅ |
| Constant response (`Shift`) | ✅ | ✅ | ✅ |
| `Spline`: `CompactSpline` | ✅ | ✅ | ✅ |
| `Spline`: `UniformSpline` | ✅ | ✅ | ✅ |
| `Spline`: `MonotonicSpline` | ✅ | ✅ | ✅ |
| `Spline`: `GeneralSpline` | ✅ | ✅ | ✅ |
| `Graph` | ✅ | ✅ | ✅ |
| `Surface` / `Bilinear` | ✅ | ✅ | ⚠️ Weights/fixed bins implemented; CUDA device validation pending |
| `Surface` / `Bicubic` | ✅ | ✅ | ⚠️ Weights/fixed bins implemented; CUDA device validation pending |
| `ExternalWeight` (`ExternalWeightDispatcher`) | ✅ | ✅ | 🚧 |
| ROOT objects (`RootFormula`, `RootSpline`, `RootGraph`) | ❌ Heavy C++ objects, unsuitable for the flat backend model | ❌ Strictly incompatible with Metal device execution as C++ ROOT objects | ❌ Rejected by `EngineLayout` |
| `Tabulated` | ⚠️ Table updated on host at each propagation | ❌ | 🚧 |
| `Kriged` | ⚠️ Table updated on host at each propagation | ❌ | 🚧 |
| `CompiledLibDial` | ❌ | ❌ | ❌ Rejected by `EngineLayout` |
| `Polynomial` | ❌ | ❌ | ❌ Rejected by `EngineLayout` |

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

| Feature | CPU | MPS | CUDA |
|---|---|---|---|
| Event weights and histogram sums / squared sums | ✅ | ✅ | ✅ Scalar dials / fixed bins |
| Statistical likelihood | ⚠️ on host | ⚠️ on host | ⚠️ On host after histogram readback |
| Separate `SampleLikelihoods` output | ❌ Only the total is stored | ❌ Only the total is stored | ❌ Only the total is stored |
| Parameter inputs and per-input mirroring | ✅ | ✅ | ✅ Scalar dials / fixed bins |
| Two-input surfaces (`Bilinear`, `Bicubic`) | ✅ | ✅ | ⚠️ Weights/fixed bins implemented; CUDA device validation pending |
| Dial extrapolation | ✅ | ✅ | ✅ Scalar dials / fixed bins |
| Hard response limits and response smoothing | ✅ | ✅ | ✅ Scalar dials / fixed bins |
| Event reweight cap | ✅ | ✅ | ✅ Scalar dials / fixed bins |
| Observable responses (`applyToVariable`) | ✅ | ✅ | 🚧 |
| Dynamic binning / event migration | ✅ | ✅ | 🚧 |
| Smoothed binning / fractional contributions | ✅ | ✅ | 🚧 |
| External-weight updates | ⚠️ Produced on host | ⚠️ Produced on host, copied to GPU buffers | 🚧 |
| Event variables as ordinary dial inputs (`DialInputSource::EventVariable`) | ❌ | ❌ | 🚧 |
| Numerical precision | Double (64-bit) | ⚠️ ~48 significant bits in resident event propagation (float pairs); float32 weights / histograms in packed execution | Double (64-bit); CPU comparisons passed on V100 |
| Hardware requirements | ✅ Host CPU | ⚠️ Apple GPU with Metal | ⚠️ NVIDIA GPU and CUDA toolkit; CC V100 smoke test uses CUDA 12.4 / `sm_70` |

The only MPS backend configuration option is `propagationMode`: `cachedResponses`
reuses eligible responses between propagations, while `recomputeResponses`
recalculates them at every propagation. Sharing responses within a propagation
is automatic. MPS selects its internal executor according to the model; this
does not add support for dial classes rejected by `EngineLayout`. See
[MPS response propagation modes](mps/BackendStructure.md#response-propagation-mode).
Compact GPU descriptors, ordered spline payloads and the resident binning
optimizations are production behavior; temporary A/B comparison switches have
been removed. GPU stage profiling remains available for performance analysis.

The compatibility rules are implemented in
[`EngineLayout.cpp`](../../src/Backends/Common/src/EngineLayout.cpp),
[`CpuBackend.cpp`](../../src/Backends/Implementations/CPU/src/CpuBackend.cpp),
[`MpsBackend.mm`](../../src/Backends/Implementations/MPS/src/MpsBackend.mm) and
[`CudaBackend.cpp`](../../src/Backends/Implementations/CUDA/src/CudaBackend.cpp).
CUDA resource management is isolated in
[`CudaDeviceContext.cu`](../../src/Backends/Implementations/CUDA/src/CudaDeviceContext.cu).
Common descriptor and numerical-semantics code is available for future CUDA
kernels, but its presence does not constitute CUDA execution support.

## Connection to the engine

`BackendManager` connects the likelihood interface to a backend: it prepares
runtime inputs, requests propagation and materializes the outputs needed by the
fitter or host-side consumers.

## Fit monitor

The monitor always displays the execution path, its iteration rate, the latest
`Propagation + LLH` duration and the minimizer time. The duration measures the
complete engine likelihood evaluation on the host, including propagation,
required transfers and likelihood calculation; it is not GPU kernel time.

Additional columns come from `getMonitorColumns()`, which returns title/value
pairs. `FitterEngine` delegates to the active backend or the original `Propagator`.
The base backend returns no additional columns. CPU provides its measured stage
times; MPS uses the default aggregate display. The original propagator provides
its own dial, reweighting and histogram or CacheManager timings. The table builder
does not depend on individual backend types.

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
