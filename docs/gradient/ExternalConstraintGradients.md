# External constraint gradients

External Gaussian constraints contribute to the total analytic gradient when
`useAnalyticGradient: true`. They use the existing `externalConstraints`
configuration; no extra gradient option is needed.

For a measurement of a formula `f(θ)` in physical parameter coordinates:

```text
penalty = ((f(θ) - mean) / stdDev)²
∂penalty/∂θ_p = 2 * (f(θ) - mean) / stdDev² * ∂f/∂θ_p
```

| Stage | Implementation |
| --- | --- |
| Formula differentiation | `FormulaEvaluator` uses ROOT/Clad, as for formula dials. |
| Alias mapping | Derivatives follow declared aliases, independently of ROOT's internal parameter order. |
| Repeated physical parameter | Contributions from all aliases referring to that parameter are summed. |
| Multiple measurements | Each enabled constraint adds its own gradient. |
| Fit coordinates | Contributions are added in physical coordinates before the eigen projection and minimizer normalization. |

Scalar formula evaluation is unchanged. A separate formula evaluator generates
and caches the derivative when analytic-gradient support is validated. Derivative
calls use its existing shared lock. A gradient request does not update the
stored scalar LLH values.

## Compatibility and limitations

- Only the currently implemented external constraint type, Gaussian, is covered.
- ROOT must support Clad differentiation for the configured formula. Unsupported
  generation or a numerical fallback causes an initialization error when analytic
  gradients are enabled, including the constraint name and expression. Value-only
  fits do not require derivative generation.
- Formula aliases refer to physical parameter values. There are no dial-response
  limits, parameter mirrors, or event variables inside this independent measurement.
- Flat priors, disabled internal penalties, and fixed status do not remove the
  external measurement. Derivative accumulation fills only the parameter entries
  requested by the caller; the minimizer retains its existing selection of fit
  parameters.
- Disabled constraints are skipped, and constant constraints have zero derivatives.
- Non-finite residuals or requested formula derivatives raise an explicit error.
  This does not regularize singular expressions or make piecewise formulas smooth.

## Regression coverage

`unitTest_ExternalConstraintGradient.cpp` covers nonlinear derivatives, cross-set
references, alias ordering and repeated parameters, subsets, parameter flags,
constant/disabled/multiple measurements, reconfiguration, unsupported generation,
and singular expressions.

`596NumericsExternalConstraintGradient.py` compares the complete LLH gradient
against finite differences in physical, normalized and eigen coordinates, with
one and four propagation threads. It also covers flat priors, fixed-parameter
requests and initialization-time rejection of unsupported external formulas.
