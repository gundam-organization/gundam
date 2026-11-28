# Resident statistical likelihoods

The CPU, CUDA and MPS backends have portable evaluators for `PoissonLLH`,
`BarlowLLH_BANFF_OA2021`, `BarlowBeestonLiteGaussian` and
`BarlowBeestonLitePoisson`. The BANFF name selects the validated
`BarlowBeestonBanff2022` class. The exact `BarlowBeeston` model and the SFGD
variant are not portable likelihoods. No extra configuration key is needed;
the existing `jointProbabilityConfig.type` selects the implementation.

For these functions the GPU pipeline evaluates the likelihood from resident
histogram sums and squared sums, reduces contributions on device, and reads
only the statistical total: one double on CUDA, one compensated float pair on
Metal (8 bytes in either case). Weights, histograms and event observables are
materialized only when requested. The existing fitter disables automatic
materialization during minimization and restores it for final output. Parameter
updates, external weight producers and existing propagation control/diagnostic
metadata still involve the host. Parameter constraints remain on the host.

## Reference implementation and descriptors

The existing JointProbability classes, including their mathematical code, are
unchanged. `snapshotJointProbability()` is a separate read-only adapter defined
in their existing translation unit (some historical headers contain non-inline
method definitions and cannot be included in another translation unit).
`EngineLayout` exports this snapshot to a passive likelihood descriptor.

`LikelihoodTypes.h` defines the wire format; `BackendLikelihood.h` implements
independent scalar mathematics. CPU and CUDA use binary64. Metal uses the
existing compensated `DoubleFloat` representation and logarithm/square-root
operations. Poisson uses an equivalent series close to its minimum to avoid
cancellation; no change is made to the original Poisson reference.

The Barlow implementation carries `usePoissonLikelihood`, `BBNoUpdateWeights`,
`fractionalErrorLimit`, `expectedValueMinimum` and `throwIfInfLlh`. Nominal
variances use an already captured reference nominal histogram if available,
otherwise the histogram at backend construction. A backend snapshot must be
rebuilt when its likelihood configuration or data change. Bin masks preserve
`ignoreBinsWithZeroPredictionAtPrior` from the existing layout construction.

The backend preserves the reference's error-to-variance conversion and the
compatibility threshold that can return infinity even with `throwIfInfLlh`.
Numerical failures requiring an exception use a NaN total and are reported on
the host without histogram readback. NaN totals are always rejected. Per-bin
verbose messages and warnings from the reference classes are not reproduced
by device kernels.

The two lite models always use the current histogram variance; they do not
capture nominal errors or export any BANFF compatibility flags. Gaussian uses
Conway's analytic profile. Poisson uses the improved Dembinski-Abdelmotteleb
effective-count model, equations (17)-(18). Both share the same portable
implementation across CPU, CUDA and Metal, including the stable near-minimum
expansions. Portable `evaluateLiteGaussian` and `evaluateLitePoisson` return
both the value and analytic partial derivatives with respect to prediction
and standard deviation. These match the host reference derivatives; the
resident likelihood kernels consume the value only. This change does not add
a new device parameter-gradient pipeline.

Invalid lite inputs return positive infinity rather than the legacy BANFF
NaN/throw sentinel. Empty data/MC bins contribute zero. The Poisson lite model
rejects zero mean with nonzero variance, matching its host reference.

The exact Barlow-Beeston solver requires counts and scale factors per source.
The resident histograms contain only sums of weights and squared weights, so
they cannot represent that model without a source decomposition. It remains
unsupported; no effective single-source replacement is introduced.

Unsupported likelihoods retain the host callback path and histogram readback.
Diagnostics expose `likelihoodExecution` as `device` or `host`. Metal selects
its compensated resident event executor for supported likelihoods in both
`cachedResponses` and `recomputeResponses` modes. `SampleLikelihoods` remains
unimplemented.

## MPS work distribution and reuse

The resident Metal executor uses one bin per thread up to 8,192 likelihood bins,
and short chunks of eight bins above that size. A fixed 128-thread tree reduces
partials on device, avoiding the previous serial final loop. `LikelihoodWorkLayout`
is shared between host and shader, so dispatch/allocation use the same plan.

Histograms and statistical likelihood are reused when the previous result is
valid and neither weight dependencies nor event-state/binning inputs changed.
External-weight generations, observable-input revisions, host edits and forced
binning updates still invalidate the prediction. Recompute mode continues to
recompute. Even on reuse, only the 8-byte total is copied back; output
materialization remains explicit. Diagnostics report `updatedPrediction`,
`likelihoodBinsPerThread`, `likelihoodPartialCount` and `likelihoodReductionWidth`.

The original `JointProbability` classes and all scalar formulas are unchanged
by this optimization. See [the optimization validation](ResidentLikelihoodOptimization.md)
for production measurements and precision tests.

## Verification and limits

`unitTest_BackendLikelihood.cpp` compares portable CPU evaluations with the
untouched references, including near-minimum values, empty bins, fractional
limits, nominal errors and failure policies. Device tests cover both modes,
multiple samples/reduction chunks, ignored bins, configuration options, stale
tokens, explicit materialization, and zero histogram/weight readback. Additional
Metal cases cover infinity, empty likelihoods and a rebuild to host fallback.
The same resident tests are compiled into the CUDA device test executable.

`BackendFitCovariance.DeviceStatLikelihoodMatchesOriginalFitAndHesse` compares
full Poisson and Barlow fits, fitted parameters, errors and Hesse covariance
against the original engine for both Metal modes.

Local validation on Apple: full debug build, 244 fast C++ tests, and the three
BackendFitCovariance tests pass. CUDA device tests are registered as
`cuda-backend-likelihood`; CUDA compilation and execution still require
validation on a CUDA machine.

Metal has the exponent range of float32 despite its compensated precision.
Packing rejects finite data/nominal variances outside its normal exponent range;
large default fractional-error limits are represented as infinity. The binary64
floor used by Barlow is represented by zero with an explicit `log(DBL_MIN)` for
the empty-prediction branch. Extreme underflow/overflow cases are not guaranteed
to reproduce binary64 behavior. The test results validate ordinary fit ranges,
not the full binary64 exponent range. CUDA uses binary64 throughout.

`BackendTimingSummary` exposes `likelihoodReadbackBytes`,
`likelihoodHostSeconds` and `likelihoodDeviceSeconds` in Python as well as C++.
CUDA measures the likelihood GPU stage; Metal reports its separate stage time
only with diagnostic stage profiling enabled (otherwise it is included in total
GPU propagation time). No performance benchmark is claimed by these tests.

## Lite model validation

`BackendLikelihood.LiteSnapshotsAndPortableValuesAndGradientsMatchReferences`
checks values, analytic partial derivatives, invalid inputs and the variance
conversion against the two reference classes. An EngineLayout test checks
selection through `jointProbabilityConfig.type`, including the absence of
nominal-error settings. CPU tests deliberately install a throwing host callback
to verify that the portable evaluator is used.

The resident integration tests cover both lite models on Metal in
`cachedResponses` and `recomputeResponses` modes, multiple samples and masked
bins, and assert zero histogram/event-weight readback. Low-rate tests cover
rates down to 1e-12, small displacements from the minimum, empty bins and
infinite likelihoods. The same resident tests are registered for CUDA builds.

Local validation for this addition: 231 fast C++ tests pass on Apple, including
live Metal execution. CUDA uses the shared host/device scalar implementation;
CUDA compilation and device execution have not been validated on this machine.
