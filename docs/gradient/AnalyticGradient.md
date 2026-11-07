# Analytic gradient

The gradient `∂LLH/∂θ` describes how the likelihood changes with each parameter.
**Analytic** means that derivatives are computed from the model formulas rather
than by numerically shifting parameters. The goal is to reduce the cost of the
fit; fewer iterations are not guaranteed.

## Calculation chain

| Block | Role |
| --- | --- |
| Dials → events | Differentiate responses and their product to obtain `∂w/∂θ`, including zero responses. |
| Events → histograms | Propagate derivatives to `μ = Σw` and `σ = √Σw²`, keeping observed data fixed. |
| Histograms → LLH | Combine `∂LLH/∂μ` and `∂LLH/∂σ`, including Barlow–Beeston profiling. |
| Penalties → total gradient | Add Gaussian constraints and transform to fit coordinates. |
| Minuit2 / MIGRAD | Supply the LLH value and full gradient through `GradFunctor`. |

## Compatibility

✅ Supported · ⚠️ Supported with restrictions · ❌ Not implemented

### Dials

| Feature | Status | Comments |
| --- | :---: | --- |
| `Norm`, `Shift`, `Graph`, `Polynomial` | ✅ | |
| `CompactSpline`, `UniformSpline`, `MonotonicSpline`, `GeneralSpline` | ✅ | |
| `Bilinear`, `Bicubic` | ✅ | |
| `RootGraph` | ✅ | Right-hand slope at interior knots; last segment at the final knot when extrapolation is enabled. |
| `RootSpline` | ✅ | |
| `Formula` | ⚠️ | Requires ROOT/Clad automatic differentiation. `eventVariable` inputs remain fixed when differentiating fit parameters. Unsupported generation or a numerical fallback raises an initialization error. Derivative calls share a lock per formula. |
| `CompositeDial` | ✅️ | All nested dials must support analytic gradients. Event inputs remain fixed. |
| `CompiledLibDial` | ❌ | |
| `Tabulated` | ❌ | |
| `Kriged` | ❌ | |
| `ExternalWeight` | ✅️ | Worker must declare gradient support and fill a dense derivative matrix per event/bin. See [worker API](../configuration/ExternalWeightDials.md#analytic-gradients). |
| Weight products | ✅ | Handles zero responses and repeated parameters. The derivative is zero when the global reweight cap is exceeded. |
| Response smoothing | ✅ | Applies the supervisor derivative through the chain rule, including nested composite nodes. |
| Mirrors, bounds and interpolation knots | ⚠️ | Piecewise conventions; derivatives may not be unique at discontinuities in slope. |

### Events → histograms

| Feature | Status | Details / remaining work |
| --- | :---: | --- |
| Fixed-bin histograms | ✅ | Bin contents, MC uncertainties and ignored bins. |
| Dynamic binning / smoothing / migration | ❌ | Differentiate bin assignments and contributions. |
| Dials modifying observables | ❌ | Propagate observable derivatives. |

### Histograms → LLH

| Feature | Status | Details / remaining work |
| --- | :---: | --- |
| `BarlowBeestonLiteGaussian` | ✅ | Derivatives with respect to bin contents and MC uncertainties. |
| `BarlowBeestonLitePoisson` | ✅ | Includes the dependence of the effective MC event count. |
| `PoissonLogLikelihood` / `PoissonLLH`, `LeastSquares` | ✅ | Analytic derivatives per bin. |
| Other LLHs | ❌ | Implement and declare analytic derivatives. |

### Penalties → total gradient

| Feature | Status | Details / remaining work |
| --- | :---: | --- |
| Correlated Gaussian penalties | ✅ | Parameter-set constraints. |
| External constraints | ❌ | Add their gradient contributions. |
| Physical, normalized and eigen coordinates | ✅ | Transform the gradient to fit coordinates. |

### Minuit2 / MIGRAD

| Feature | Status | Details / remaining work |
| --- | :---: | --- |
| Minuit2 / MIGRAD | ✅ | Value and gradient supplied; Simplex does not use derivatives. |
| G2 / analytic Hessian | ❌ | Only first derivatives are supplied. |
| Shared value + gradient cache | ❌ | Each gradient request propagates the parameters again. |

## Activation

```yaml
fitterEngineConfig:
  minimizerConfig:
    useAnalyticGradient: true
```

Disabled by default. An incompatible configuration causes an **explicit error
at initialization**, with no automatic fallback to numerical gradients.
The log reports `Supplied gradient calls`.

Comparisons against finite differences are covered in particular by
[`590NumericsWeightGradient.py`](../../tests/python/fast/590NumericsWeightGradient.py).
