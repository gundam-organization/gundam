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

## Common libraries

`src/Backends/Common` is split into five libraries, each with its own
`include`, `src` and `CMakeLists.txt`:

| Target | Responsibility | Dependencies within Common |
| --- | --- | --- |
| `GundamBackendsCommonFormula` | Formula compilation and arithmetic evaluation | None |
| `GundamBackendsCommonBinning` | Bin geometry, spatial index and binning semantics | None |
| `GundamBackendsCommonModel` | Backend contract, engine views and propagation inputs/outputs | Formula, Binning |
| `GundamBackendsCommonExecution` | Shared evaluation, packing and cache invalidation | Model |
| `GundamBackendsCommonManager` | Factory and engine integration | Model |

Manager is the entry point. Its factory links the enabled implementation libraries:
`GundamBackendsCPU` (always), `GundamBackendsCUDA` (when CUDA is enabled), and
`GundamBackendsMPS` (on macOS). Each implementation has its own `CMakeLists.txt`
under `Implementations` and depends on Execution. The abstract `Backend` contract
belongs to Model so implementations do not depend on Manager. `GundamBackends`
remains a CMake alias for Manager. Only Manager depends on `GundamStatisticalInference`; the other
libraries operate on backend descriptors. Conversion from `Histogram` to
`BinningGeometry` belongs to Manager's `BinningGeometryBuilder`.

`WITH_BACKENDS` enables the whole group. Each library follows `USE_STATIC_LINKS`
like the other GUNDAM libraries: shared libraries by default, static archives
when enabled. Library suffixes follow the platform (`.so` on Linux, `.dylib`
on macOS).

## Available backends

Implementation details, numerical precision and performance choices are documented
with each backend:

- [Backend activation and configuration](BackendLifecycle.md).
- [CPU structure](cpu/BackendStructure.md) and [optimization choices](cpu/ReweightOptimization.md).
- [MPS structure](mps/BackendStructure.md).
- [CUDA implementation and validation](../guides/CudaBackend.md).
- [Multi-input dials](MultiInputDials.md).
- [Small-fit and Hesse covariance regression](BackendFitCovariance.md).
- [CUDA production profiling](../../tests/python/benchmarks/README.md#cuda-production-profiling).
- [CUDA OA2024 profiling results and optimization priorities](CudaOa2024Benchmark.md).

## Compatibility

These tables describe the `CPU`, `MPS` and `CUDA` backends selected through
`backendManagerConfig.useBackend`. The original propagation path and the legacy
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
| Host materialization | ✅ Event state, weights and histograms | ✅ Event state, weight and histogram readback | ✅ Lazy weight, histogram and event-state readback; strict path validated at CC |
| Dial-response and event-state caches | ✅ | ✅ Depends on propagation mode | ✅ Sparse response cache validated on CC OA2024; strict observable dependency/revision cache validated at CC |
| Backend activation/deactivation | ✅ Fixed, dynamic and smoothed binning | ✅ Dynamic/smoothed lifecycle tested against CPU and original engine | ⚠️ Same lifecycle API; device lifecycle regression must run at CC with `GUNDAM_TEST_CUDA=1` |
| Numerical regression tests | ✅ CPU comparisons | ✅ MPS comparisons | ✅ Both modes compared against CPU on CC V100; error/retry tests passed |
| Small fit and full Hesse covariance regression | ✅ Original/CPU comparison and analytic covariance reference pass locally | — Not included in this regression | ✅ Both modes pass at CC, including the analytic covariance reference |
| Standalone CUDA infrastructure test | — | — | ✅ FP64 kernel validated on CC-IN2P3 V100, interactive and Slurm batch |

CUDA implements scalar, Surface, external and table dials for event weights.
Observable responses and strict event migration passed CC validation on 2026-10-25
(five CUDA CTest suites, 1.53 s total). Smoothed binning is implemented and
has now been confirmed working at the CC (2026-10-25).
Its defaults are `propagationMode: cachedResponses` and
`weightLayout: warpInterleaved`. Index padding above 2x retains `eventMajor`.
`cachedResponses` (default) retains dial responses until any transformed input
changes; `recomputeResponses` evaluates every unique dial on every propagation.
Both modes recompute event weights and histograms, using native doubles.
`ExternalWeightDispatcher`, `Tabulated` and `Kriged` responses are implemented
for event weights with fixed bins; their producers still run on the host.
Observable responses compose Override, Multiply and Add on device and expose
lazy event-state readback. Smoothed binning adds fractional and outside
contributions on device; ordinary event-variable dial inputs remain rejected
during model preparation. No CPU propagation fallback is performed.

Bilinear/Bicubic surfaces use the shared interpolation and retain ordered x/y
inputs, per-input mirroring, extrapolation, response bounds and smoothing.
Disjoint dependency-pair groups prevent duplicate response writes when both
parameters change. Scalar-only models retain the specialized scalar kernels;
mixed models allocate the additional second-input cache only for surfaces.
Host packing/validation tests pass locally. `cuda-backend-surfaces` compares
against native dials and CPU and passed at CC-IN2P3 on 2026-10-24 together with
`cuda-backend-skeleton` and `cuda-backend-propagation` (3/3 CTest suites).

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
with `Unsupported dial type for EngineView`. There is no automatic fallback to
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
| `Surface` / `Bilinear` | ✅ | ✅ | ✅ |
| `Surface` / `Bicubic` | ✅ | ✅ | ✅ |
| `ExternalWeight` (`ExternalWeightDispatcher`) | ✅ | ✅ | ✅ |
| ROOT objects (`RootFormula`, `RootSpline`, `RootGraph`) | ❌ Heavy C++ objects, unsuitable for the flat backend model | ❌ Strictly incompatible with Metal device execution as C++ ROOT objects | ❌ Rejected by `EngineLayout` |
| `Tabulated` | ⚠️ Table updated on host at each propagation | ⚠️ Host table, GPU response | ⚠️ Host table, GPU response; weights validated at CC |
| `Kriged` | ⚠️ Table updated on host at each propagation | ⚠️ Host table, GPU response | ⚠️ Host table, GPU response; weights validated at CC |
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

MPS and CUDA use the same external block layout for dispatchers and tables.
MPS retains compensated float pairs for table values, terms and responses.
Only changed generations are uploaded (native doubles on CUDA); disjoint per-block
response groups are evaluated independently of parameter changes. Immutable
indices and coefficients retain native term order, including negative, repeated
and unsorted Kriged terms. `cachedResponses` reuses published responses;
`recomputeResponses` reevaluates all responses while still avoiding unchanged
block transfers. Errors and rebuilds force fresh uploads and responses.
`BackendManager` republishes host tables at every propagation because these
factories do not expose producer revisions; consequently those tables are
uploaded every call, including at unchanged parameters.

Host validation and native/CPU Python regressions pass locally. CC validation
of external/table event weights was confirmed on 2026-10-25. Opt-in CUDA runs
of `710BackendsMultipleExternalWeight.py` / `762BackendsCudaTableDials.py` remain available for
end-to-end checks. See [CUDA validation](../guides/CudaBackend.md).

### Engine features

Here, ✅ means the backend can handle the feature; it does not imply that
every stage runs on the GPU. MPS support applies to both `cachedResponses` and
`recomputeResponses` unless indicated otherwise.

| Feature | CPU | MPS | CUDA |
|---|---|---|---|
| Event weights and histogram sums / squared sums | ✅ | ✅ | ✅ |
| Statistical likelihood | ⚠️ on host | ⚠️ on host | ⚠️ on host |
| Separate `SampleLikelihoods` output | ❌ Only the total is stored | ❌ Only the total is stored | ❌ Only the total is stored |
| Parameter inputs and per-input mirroring | ✅ | ✅ | ✅ |
| Two-input surfaces (`Bilinear`, `Bicubic`) | ✅ | ✅ | ✅ |
| Dial extrapolation | ✅ | ✅ | ✅ |
| Hard response limits and response smoothing | ✅ | ✅ | ✅ |
| Event reweight cap | ✅ | ✅ | ✅ |
| Observable responses (`applyToVariable`) | ✅ | ✅ | ✅ |
| Dynamic binning / event migration | ✅ | ✅ | ✅ |
| Smoothed binning / fractional contributions | ✅ | ✅ | ✅ |
| External-weight updates | ⚠️ Produced on host | ⚠️ Produced on host, copied to GPU buffers | ⚠️ Produced on host; changed blocks copied as doubles; weights validated at CC |
| Event variables as ordinary dial inputs (`DialInputSource::EventVariable`) | ❌ | ❌ | 🚧 |
| Numerical precision | Double (64-bit) | ⚠️ ~48 significant bits in resident event propagation (float pairs); float32 weights / histograms in packed execution | Double (64-bit); CPU comparisons passed on V100 |
| Hardware requirements | ✅ Host CPU | ⚠️ Apple GPU with Metal | ⚠️ NVIDIA GPU and CUDA toolkit; CC V100 smoke test uses CUDA 12.4 / `sm_70` |

The CUDA observable path reuses the response slots used for event weights,
including surfaces, external publications and tables. Event state is invalidated
by relevant parameter values, producer generations, input revisions or host edits.
The current strict bin search preserves the first matching bin (including
zero-width selections), then a stable GPU sort rebuilds the histogram index in
original event order. Weight-only changes retain the resident observable state
and index. The smoothed path counts and emits compact per-event contributions,
including the outside remainder, then stably groups them by global bin. Histogram
squared sums use `(weight * fraction)^2`. Its buffers grow with actual
contributions rather than an events-by-bins matrix. Revisions and fractional
state are exposed through `EventState`; weight-only changes reuse this index.
Host math/validation and CPU/MPS regressions pass; GPU execution was confirmed
at the CC on 2026-10-25. See [CUDA observable responses](CudaObservableResponses.md) and
[smoothed binning validation](CudaSmoothedBinning.md). Production workload commands
are in the [CUDA migration/smoothing benchmark](../../tests/python/benchmarks/README.md#cuda-migration-and-smoothing-on-upgrade-cc-2026).
The [upgrade-cc-2026 production results](CudaUpgradeBinningBenchmark.md) pass
CPU comparisons for Pmu migration with 1 MeV smoothing: cached CUDA takes
4.10 ms per varying-parameter evaluation versus 29.93 ms on four CPU threads
(7.30 times faster). Observable/search and fractional index construction account
for 93.3% of GPU propagation time in that workload.
An ordered bin hierarchy now prunes hard/fractional searches, and diagnostic
counters aggregate per warp. The CC stress production run passes CPU comparisons
and reduces cached evaluations from 4.08 to 2.92 ms (28.3% less time), versus
30.04 ms on four CPU threads (10.28 times faster). See the
[stress baseline and indexed results](CudaUpgradeBinningBenchmark.md#stress-baseline-and-indexed-search-retest).

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
[`EngineLayout.cpp`](../../src/Backends/Common/Manager/src/EngineLayout.cpp),
[`CpuBackend.cpp`](../../src/Backends/Implementations/CPU/src/CpuBackend.cpp),
[`MpsBackend.mm`](../../src/Backends/Implementations/MPS/src/MpsBackend.mm) and
[`CudaBackend.cpp`](../../src/Backends/Implementations/CUDA/src/CudaBackend.cpp).
CUDA resource management is isolated in
[`CudaDeviceContext.cu`](../../src/Backends/Implementations/CUDA/src/CudaDeviceContext.cu).
Common descriptor and numerical-semantics code is available for future CUDA
kernels, but its presence does not constitute CUDA execution support.

## Analytic gradients

Gradient support as of 2026-11-11. These tables describe the **backend gradient
path**, independently of the value-propagation tables above. Features available
in the [original engine's analytic gradient](../gradient/AnalyticGradient.md)
are not automatically available in a backend.

✅ Implemented · ⚠️ Supported with restrictions · ❌ Not implemented/supported.
These statuses describe the code, not a new validation run on each device.

### Dials → event weights

| Feature | CPU | MPS | CUDA | Comments |
| --- | :---: | :---: | :---: | --- |
| `Norm` | ✅ | ✅ | ✅ | Single parameter input. |
| `Shift` | ✅ | ✅ | ✅ | |
| `Graph` | ✅ | ✅ | ✅ | Single parameter input. |
| `CompactSpline`, `UniformSpline`, `MonotonicSpline`, `GeneralSpline` | ✅ | ✅ | ✅ | Single parameter input. |
| `Bilinear`, `Bicubic` | ✅ | ✅ | ✅ | CPU, MPS and CUDA differentiate both inputs, including shared parameters, mirroring and clamps. |
| `RootGraph`, `RootSpline` | ❌ | ❌ | ❌ | Original-engine support does not extend to the backend gradient path. |
| `Formula` (`RootFormula`, `EventFormula`) | ✅ | ✅ | ✅ | CPU, MPS and CUDA differentiate compiled bytecode, including named inputs, bound event constants, mirrors and response limits; no ROOT/Clad gradient generation. |
| `CompositeDial` | ✅ | ✅ | ✅ | CPU, MPS and CUDA differentiate compiled graphs, including formula nodes, surfaces, branches and nested composites. |
| `Polynomial` | ❌ | ❌ | ❌ | No backend gradient implementation. |
| `Tabulated`, `Kriged` | ✅️ | ✅ | ✅ | CPU, MPS and CUDA consume producer Jacobians; requires a configured `gradientFunction`. |
| `ExternalWeightDispatcher` | ✅️ | ✅ | ✅ | CPU, MPS and CUDA consume current worker gradients; the worker must support the gradient protocol. |
| `CompiledLibDial` | ❌ | ❌ | ❌ | No backend gradient implementation. |
| Weight products | ✅ | ✅ | ✅ | Handles zero responses and repeated parameter occurrences without division. |
| Parameter mirroring | ⚠️ | ⚠️ | ⚠️ | Requires a positive finite mirror range; piecewise derivative conventions. |
| Extrapolation and hard response limits | ✅ | ✅ | ✅ | Piecewise conventions at knots and limits. |
| Response smoothing | ✅ | ✅ | ✅ | CPU, MPS and CUDA apply the transition derivative to every supported dial, composite node and composite output. |
| Global event reweight cap | ✅ | ✅ | ✅ | Weight derivative is zero above the cap. |

CPU, MPS and CUDA response smoothing use the same transition thresholds and widths as value
propagation. Within each transition the local derivative is multiplied by
`t²(3 − 2t)`; it is zero beyond the outer edge and one on the identity side.
Hard limits retain their existing piecewise conventions when widths are zero.
This applies to formula and surface derivatives, producer Jacobians, and every
level of nested composites. Histogram/bin smoothing is differentiated separately
as described below.

CPU, MPS and CUDA Formula gradients apply the chain rule directly to compiled bytecode,
for standalone dials and composite nodes. All supported formula opcodes have
derivative rules. `floor`/`ceil` use zero piecewise derivatives, `abs` uses zero
at zero, and `min`/`max` select the left operand at ties. Host-side formula
construction/binding can still use ROOT; backend dial validation and numerical
gradient evaluation do not request ROOT/Clad differentiation.

### Events → histograms → statistical LLH

| Feature | CPU | MPS | CUDA | Comments |
| --- | :---: | :---: | :---: | --- |
| Fixed, unsmoothed histograms | ✅ | ✅ | ✅ | Derivatives of bin contents and MC uncertainties. |
| Ignored likelihood bins | ✅ | ✅ | ✅ | Follow the prepared likelihood's ignored-bin mask. |
| Observable-response dials + dynamic binning / migration + smoothed binning / fractional contributions | ⚠️ | ⚠️ | ⚠️ | CPU, MPS and CUDA differentiate Override/Multiply/Add, fractions and `(weight * fraction)^2`. Moving histogram axes require all three features, with smoothing on every moving axis; hard migration is rejected. |
| `BarlowBeestonLiteGaussian` | ✅ | ✅ | ✅ | Includes derivatives with respect to prediction and MC uncertainty. |
| `PoissonLLH` / `PoissonLogLikelihood` | ❌ | ❌ | ❌ | Value support does not imply gradient support. |
| `BarlowBeestonLitePoisson` | ❌ | ❌ | ❌ | No backend gradient implementation. |
| `LeastSquares` and other statistical LLHs | ❌ | ❌ | ❌ | Gradient validation currently requires Gaussian Barlow–Beeston. |

CPU, MPS and CUDA migration gradients combine the dial Jacobians with the observable
composition and the compact-support binning kernel. Every moving histogram axis
must have both dynamic binning and smoothing; smoothing an unrelated axis does
not satisfy this requirement. Hard selections on parameter-independent axes
remain supported. Observable responses outside the histogram coordinates have
no statistical derivative. Fixed fractional histograms without observable dials
still contribute through their event-weight derivatives.

For each bin contribution `w * f`, CPU, MPS and CUDA differentiate both the event weight `w`
and the fraction `f`, including their effect on the MC squared-weight sum.
Zero/repeated observable multipliers use products without division. Events
outside the strict bins can still contribute within the smoothing support; the
outside remainder is excluded from the statistical LLH. The small roundoff
correction to the largest fraction is also applied to its Jacobian. A global
weight cap suppresses only the weight derivative, retaining observable migration.
The existing Gaussian Barlow–Beeston likelihood restriction still applies.

### Penalties → fit coordinates → minimizer

These stages belong to the shared host-side integration rather than the device
kernels. Their status applies when using the corresponding backend.

| Feature | CPU | MPS | CUDA | Comments |
| --- | :---: | :---: | :---: | --- |
| Internal correlated Gaussian penalties | ✅ | ✅ | ✅ | Added through `LikelihoodInterface::finalizeGradient()`. |
| Physical, normalized and eigen coordinates | ✅ | ✅ | ✅ | Shared parameter bindings and fit-coordinate transformations. |
| External Gaussian constraints | ❌ | ❌ | ❌ | Their scalar penalty is evaluated, but their physical derivatives are not added by `BackendManager::evalGradient()`. |
| Minuit2 / MIGRAD supplied gradient | ✅ | ✅ | ✅ | Activated with `useAnalyticGradient: true`. |
| G2 / analytic Hessian | ❌ | ❌ | ❌ | Only first derivatives are supplied. |
| Shared value + gradient propagation | ❌ | ❌ | ❌ | Gradient requests invoke propagation independently of value requests. |

**External constraints are a current integration gap:** their formulas are
validated for differentiation, but the backend manager does not call
`ExternalConstraints::evalGradient()` before finalizing the gradient. The result
therefore omits that contribution even though the scalar LLH includes it. Do
not use the backend analytic-gradient path with enabled external constraints
until this contribution is connected; current validation does not reject this
combination automatically.

### Execution and regression coverage

| Stage | CPU | MPS | CUDA |
| --- | --- | --- | --- |
| Statistical derivative execution | Host, double precision | Resident Metal kernels, `ObservableScalar` float pairs | CUDA kernels, native double precision |
| Event derivative accumulation | Thread-local parameter sums and host reduction | Device contributions and reduction by parameter | Device contributions and reduction by parameter |
| Result returned to the host | Statistical parameter-gradient vector | Statistical parameter-gradient vector, reconstructed as doubles | Statistical parameter-gradient vector, copied as doubles |
| Model requirement | Built backend and current propagation token | Built resident event-propagation model and current token | Built device model and current token |
| Regression sources | Native backend tests and Python comparisons | Conditional native comparisons and explicit Python backend selection | Conditional device comparisons and explicit Python backend selection |

CPU, MPS and CUDA use
[`GradientValidation.h`](../../src/Backends/Common/Execution/include/GradientValidation.h)
for common structural gradient validation: statistical likelihood, dial inputs,
formula/composite bindings, mirroring and smoothing on every moving histogram
axis. Lifecycle and producer-publication checks remain in each executor; CUDA
also retains its uint32 payload-index checks.

Formula derivative rules are defined once in
[`SharedFormulaGradientOperations.h`](../../src/Backends/Common/Formula/include/DeviceMath/SharedFormulaGradientOperations.h),
using the same rule table in the CPU, CUDA and Metal interpreters. Each backend
keeps its input reader, stack and math operations, including compensated Metal
arithmetic. Zero tangents still suppress singular inactive partials; tie and
piecewise conventions are unchanged. The Metal source embeds the table at CMake
configuration time and tracks changes to it.

Manager caches the host structural gradient validation for the prepared model.
Activation/rebuild, deactivation, data-histogram updates and interface replacement
invalidate it. Structural changes to an active model require reactivation, as
for value propagation. External/table producer checks still run on every
gradient request because they also request the next Jacobian publication.

External/table gradient bindings and publications contain only producer dials,
with explicit, ordered backend dial indices. CPU copies their Jacobians and raw
responses into compact reusable snapshots. Models without producers allocate
no per-dial external Jacobian rows. The input contract still accepts the legacy
dense per-dial Jacobian format; sparse publications reject invalid, duplicate,
unsorted or non-producer indices, and missing rows cannot reuse a previous
publication. The unique-dial derivative loop remains serial and recomputes
local derivatives for every gradient request.

MPS evaluates local derivatives on the GPU in compensated float pairs. Its
immutable sparse layout has one direction per dial input and one contribution
per event/dial/input occurrence. Repeated parameter bindings reduce into the
same parameter, without division by responses or a dense event/parameter
Jacobian. Composite directions own separate scratch frames; formulas, surfaces,
edge mirrors and response smoothing are differentiated within these frames.
External/table Jacobians are copied before propagation returns, including at
unchanged parameters, and uploaded for the current gradient. Missing, malformed
or non-finite publications are rejected; an older Jacobian is never reused.
Table values and response supervision also run on the device; producers remain
on the host. Observable adjoints and fraction derivatives also run on the GPU,
using resident observables, sparse contributions and the same bin-search tree
and compact-support geometry as propagation. The event accumulation includes
both weight and observable dial occurrences; no dense event/bin or
event/parameter Jacobian is stored. Fixed bins use resident bin contributions,
so dynamic state never depends on a stale initial event bin.

CUDA dial gradients are implemented in native doubles with one analytic
forward direction per root input. An independent gradient layout retains
original dial numbering, formula bindings and full composite graphs, avoiding
coupling to value-path reordering, compact descriptors or template deduplication.
Producer Jacobians are copied per propagation and checked before gradient use;
missing, malformed or non-finite rows cannot reuse an older publication.
Constant/zero-input responses remain in event products but have no parameter
occurrences. Observable occurrences follow weight occurrences in the sparse
layout. CUDA differentiates the shared observable composition and the same
compact-support fractions and first-largest roundoff correction used by value
propagation, reading resident observables and histogram moments. The geometry
traversal uses the resident CUDA candidate tree; it stores no dense event/bin or
event/parameter Jacobian. Weight caps retain the observable migration derivative.
Hard migration and smoothing missing from any moving axis are rejected.

The CUDA derivative semantics and occurrence layout have been tested on the
host against CPU gradients and finite differences, without a CUDA toolkit. Migration rules are also compared with CPU gradients and
finite differences for multiple axes, zero/repeated multipliers, outside support,
ignored bins, producer Jacobians and capped weights.
**NVCC compilation and GPU execution of these new dial and migration gradients remain
unvalidated.** Conditional `CudaGradientDevice.*` tests exercise both propagation
modes and weight layouts in `gundamGTest_cudaBackend.exe`. Integration scripts
`416DialsExternalWeightGradient.py`, `723BackendsCpuCompositeDials.py` and
`762BackendsCudaTableDials.py` enable CUDA gradient checks with
`GUNDAM_TEST_CUDA=1`; `700BackendsCommonWeightGradient.py` accepts `--backend CUDA`.

GPU gradient evaluation reads resident weights and histogram moments; the final
parameter-gradient vector is returned to the host for penalties and minimizer
integration. It does not require transferring every event derivative to the host.

MPS regression coverage also includes opt-in integration runs of
[`416DialsExternalWeightGradient.py`](../../tests/python/fast/416DialsExternalWeightGradient.py),
[`723BackendsCpuCompositeDials.py`](../../tests/python/fast/723BackendsCpuCompositeDials.py)
and [`762BackendsCudaTableDials.py`](../../tests/python/fast/762BackendsCudaTableDials.py)
with `GUNDAM_TEST_MPS=1`. These exercise producer publication, nested composites,
response smoothing and table values/gradients in the full fitter.

Regression sources are
[`unitTest_CpuBackendGradient.cpp`](../../tests/GTests/unitTest_CpuBackendGradient.cpp)
(which also contains conditional MPS/CUDA comparisons) and
[`700BackendsCommonWeightGradient.py`](../../tests/python/fast/700BackendsCommonWeightGradient.py)
(`--backend CPU|MPS|CUDA`, default `CPU`). The Python regression compares with the
original engine and finite differences, including normalized/eigen coordinates
and backend activation/deactivation. Having these tests in the repository does
not establish that they have passed on every GPU; the CUDA propagation validation
reported elsewhere in this document is not itself gradient validation.

CPU/MPS/CUDA migration/fraction regressions are covered by
[`724BackendsCpuObservableGradient.py`](../../tests/python/fast/724BackendsCpuObservableGradient.py)
(`--backend CPU|MPS|CUDA`, default `CPU`) and native CPU/MPS plus host/device CUDA gradient tests.
They compare with the original engine and
finite differences, including absolute/relative smoothing widths, parallel
execution, normalized/eigen coordinates, zero/repeated multipliers, weight caps,
ignored bins, multidimensional geometry and outside support, plus rejection of
unsmoothed moving axes and backend activation/deactivation.

For performance measurements on existing configurations,
[`GradientPerformance.py`](../../tests/python/benchmarks/GradientPerformance.py)
accepts fitter-style `-c/-of/-t/-a` options and compares original/CPU callbacks.
It records varying/unchanged workloads, propagation stage clocks, numerical
comparisons and separate native gradient profiles. See
[usage and measurement scope](../../tests/python/benchmarks/GradientPerformance.md).

The restrictions above are enforced in
[`CpuBackendGradient.cpp`](../../src/Backends/Implementations/CPU/src/CpuBackendGradient.cpp)
and the shared device
[`GradientModel.h`](../../src/Backends/Common/Execution/include/GradientModel.h) and
[`MpsGradientModel.h`](../../src/Backends/Implementations/MPS/include/MpsGradientModel.h).
Unsupported dial, likelihood and binning combinations fail explicitly; there is
no automatic fallback to the original engine's gradient. The external-constraint
integration gap described above is an exception to this rejection behavior.

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

GPU backends add a compact line immediately before the timing table:
`Backend MPS (Apple M4 Max) / Memory: 851MB / On-device time: 51.5%`.
`Memory` reports CUDA backend buffer capacities, or Metal resources allocated
through the device in this process (including CPU/GPU shared buffers). It is
not a measurement of physical GPU residency. Memory queries run only on monitor
refresh. Free-memory and recommended-budget details are omitted from this line.

`On-device time` is the last GPU command/stream interval as a percentage of the
last backend request's wall time (including host likelihood). These intervals
include transfers and can contain idle gaps; the percentage is not hardware
utilization, occupancy, or computational efficiency. No extra GPU profiling
passes or explicit synchronization are introduced for the monitor.

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
