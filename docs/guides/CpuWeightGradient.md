# CPU weight gradients with Minuit2

Minuit2 can use the supplied CPU gradient with this opt-in configuration:

```yaml
fitterEngineConfig:
  minimizerConfig:
    useAnalyticGradient: true
  likelihoodInterfaceConfig:
    jointProbabilityConfig:
      type: BarlowBeestonLiteGaussian
    backendManagerConfig:
      isEnabled: false
```

`useAnalyticGradient` defaults to `false`, preserving numerical Minuit gradients.
With `true`, a `ROOT::Math::GradFunctor` supplies both `evalFit()` and the complete
gradient in fit coordinates. The same registration is used for a temporary
Simplex pre-fit; the main MIGRAD minimizer retains its gradient callback.
Simplex itself does not use derivatives. Other minimizer implementations are
rejected when this option is enabled.

Unsupported model checks run at initialization and throw a logged exception;
there is no automatic fallback to the value-only functor. Likelihood classes
must explicitly advertise analytic bin gradients. Currently this includes
Gaussian/Poisson Barlow-Beeston Lite, PoissonLLH and LeastSquares. Legacy
likelihoods with only finite-difference gradients are rejected. The CPU backend also supports this callback for Gaussian Barlow-Beeston Lite;
CUDA and MPS now support the same restricted gradient model; see [device gradients](BackendWeightGradient.md). The callback is available directly from Python.

After engine initialization, with the original engine or a supported backend:

```python
minimizer = engine.getMinimizer()
gradient = minimizer.evalFitGradientAt(point)
```

`point` has the same length, order and coordinates as `minimizer.evalFit(point)`.
The callback installs the point, propagates the original engine, evaluates the
likelihood and computes the total gradient. With CPU active, it instead uses
the backend propagation and statistical derivative pass. Values are returned in normalized
fit coordinates when `useNormalizedFitSpace` is enabled. Eigen-parameter
coordinates use the transpose of the physical parameter transformation.
The physical-coordinate gradients remain accessible through `Parameter.getGradient()`.
The likelihood buffers retain the evaluated total, statistical and penalty values.

The corresponding C++ callback is:

```cpp
void MinimizerBase::evalFitGradientAt(const double* parameters, double* gradient);
```

The callback is registered when `useAnalyticGradient` is true. A separate
`getNbGradientCalls()` counter on `RootMinimizer` records calls made through this
functor; `isUsingAnalyticGradient()` reports the configured mode. The fit log
reports gradient calls separately from value evaluations. Callback counts are
reset at minimizer initialization and also include requests during error calculations.
The selected option is saved with the minimizer settings.

Each gradient callback currently repropagates the point for correctness. Value
and gradient evaluations at an identical point are not cached together; lower
value-call counts alone therefore do not establish a speedup.

The event derivative pass uses the propagator's persistent worker pool. Each
worker reuses product scratch buffers and accumulates its own parameter vector;
parameter gradients are updated only after joining the workers. Parameter lookup
is built once per callback instead of searching the full fit list for every dial.
The finite-difference checks run with both one and four threads.

See [CPU gradient benchmarks](CpuGradientBenchmark.md) for a reproducible short
comparison, measured fit times and the distinction between MIGRAD and covariance
work.

## Statistical likelihood chain

For each fixed bin, `mu = sum(w)` and `sigma = sqrt(sum(w*w))`. The contribution
to each event-weight derivative is

```
dL/dw = dL/dmu + (w/sigma) * dL/dsigma
```

For `BarlowBeestonLiteGaussian`, the profiled Gaussian nuisance is evaluated
once per bin to obtain both analytic envelope derivatives. The uncertainty is
updated with the current MC weights; it is not held fixed while differentiating
the weights. Ignored bins contribute zero. Undefined/nonfinite gradients raise
an error, rather than being passed to Minuit.

The implementation includes analytic Graph, normalization, constant,
polynomial and supported spline derivatives. Bilinear/bicubic clamping retains
the derivative along the coordinate that remains inside its domain. Products
use prefix/suffix factors, including zero responses, without dividing by a dial
response. Standard parameter-set Gaussian penalties are included.

Observable-response dials, dynamic/smoothed binning, unsupported dials, enabled
external constraints are rejected. Backends validate their own supported likelihood and dial types. Graph knots, clamps and
mirror turning points are not generally differentiable; the existing piecewise
conventions apply. This does not provide derivatives of event migration.

`903CpuWeightGradient.py` checks Gaussian Barlow-Beeston with Graph/Spline
weights against finite differences in physical, normalized and eigen spaces,
including nonzero correlated penalties, a free parameter excluded from the
eigenbasis, ignored bins, stale prior state and unchanged likelihood buffers.
It checks zero-response products using LeastSquares and one-sided differences
at the normalization clamp. It also verifies unsupported model rejection.


`BackendFitCovariance.OriginalAnalyticGradientMatchesNumericalFitAndHesse`
compares Gaussian Barlow-Beeston fits with numerical and supplied gradients,
including normalized coordinates and an optional Simplex pre-fit. It checks
convergence, EDM, parameters, errors and the full Hesse covariance, and verifies
that Minuit actually calls the supplied gradient. The opt-in/default and
configuration rejection checks are in `RootMinimizerSimplex`.

## CPU backend

To use the CPU backend's resident weights and histograms:

```yaml
fitterEngineConfig:
  minimizerConfig:
    useAnalyticGradient: true
  likelihoodInterfaceConfig:
    jointProbabilityConfig:
      type: BarlowBeestonLiteGaussian
    backendManagerConfig:
      isEnabled: true
      useBackend: CPU
```

The first backend gradient implementation supports normalization, constant,
Graph, CompactSpline, UniformSpline, MonotonicSpline and GeneralSpline weights.
It includes input mirrors, extrapolation/clamping, hard response limits and the
maximum event reweight cap. Smooth response limits and complex dials are refused.
Other statistical likelihoods remain available with numerical Minuit gradients,
but are refused with analytic gradients on CPU for now.

`BackendCapabilities::supportsLikelihoodGradient` advertises the API;
`Backend::validateLikelihoodGradient()` additionally validates the current model.
`getLikelihoodGradient(token)` returns only the statistical gradient in physical
model parameter order and rejects stale tokens. The manager maps model indices
to physical parameters; `LikelihoodInterface` adds penalties and projects into
fit coordinates using the same code as the original engine. Neither the
`JointProbability` classes nor their reference implementations are modified.

The derivative pass evaluates each unique dial derivative once, reuses cached
responses where available, and accumulates event products in parallel into
per-worker parameter vectors. There is no dense event-by-parameter Jacobian.
The pass reads backend histogram buffers directly. It works with automatic
histogram/event-weight materialization disabled; callers can still request those
outputs for diagnostics. Value and gradient callbacks continue to propagate
independently, so repeated identical points are not jointly cached yet.

`903CpuWeightGradient.py` compares backend CPU, original engine and finite
differences with materialization disabled, in physical/normalized/eigen
coordinates and with one/four threads. It checks backend activation/deactivation
and incompatible analytic configurations. `CpuBackendGradient` C++ tests cover
small rates, zero responses, repeated parameter inputs, cached/uncached dials,
mirrors, extrapolation, hard response limits, the event cap, ignored bins and
stale tokens. `BackendFitCovariance.CpuAnalyticGradientMatchesOriginalNumericalFitAndHesse`
compares the final fit and Hesse covariance and verifies gradient callback use.
