## LikelihoodInterfaceConfig
[< back to parent (fitterEngineConfig)](./FitterEngine.md)

### Description

Define how the likelihood will be calculated.  This includes defining the
datasets that will be used for the model and measurement.  It also includes
defining the "propagator" that will apply the effect of fitting parameters
to the model.  The comparison between the measurement and model is defined
using `jointProbabilityConfig`.

### Config options

| Likelihood Interface Options        | Type | Descriptions                                | default   |
|-------------------------------------|------|---------------------------------------------|-----------|
| [propagatorConfig](./Propagator.md) | Json | Configure the parameter propagation         |           |
| dataSetList                         | Json | Configure the input data sets               |           |
| jointProbabilityConfig              | Json | Configure the joint probability calculation | BarlowLLH |
| plotGeneratorConfig                 | Json |                                             |           |
| enableStatThrowInToys               | bool |                                             |           |
| gaussStatThrowInToys                | bool |                                             |           |
| enableEventMcThrow                  | bool |                                             |           |

### `jointProbabilityConfig`

The `jointProbabilityConfig` block is forwarded to the selected joint
probability implementation.

#### Common options

| Joint Probability Options             | Type | Description                                                                                                                              | default |
|---------------------------------------|------|------------------------------------------------------------------------------------------------------------------------------------------|---------|
| ignoreBinsWithZeroPredictionAtPrior   | bool | If `true`, bins with a null model prediction at the prior are ignored in the likelihood evaluation. A warning is printed for each bin disabled this way. | `false` |

This option is intended for analyses where the user-defined binning contains
bins that are meant to stay empty, for example with square binning layouts
that include unreachable phase-space regions.

### `externalConstraints`

External measurements add independent contributions to the penalty likelihood.
The block is disabled by default; each constraint is enabled by default.

```yaml
externalConstraints:
  isEnabled: true
  constraintsList:
    - name: mixingAmplitude
      isEnabled: true
      parameters:
        - name: thetaValue
          parameterSetName: oscillation
          parameterName: theta
          # Alternatively use parameterIndex: 0 (the GUNDAM parameter index).
      formula: "TMath::Sin(2*[thetaValue])^2"
      type:
        name: gaussian
        mean: 0.87
        stdDev: 0.02
```

The values above are illustrative. Each `name` in `parameters` is a local formula
alias. Specify exactly one of `parameterName` and `parameterIndex`, together with
`parameterSetName`. All aliases must be unique and used by the formula; all
formula inputs must be declared aliases in brackets. Free variables such as `x`
are not accepted. Angles supplied to trigonometric functions are in radians.

Only `type.name: gaussian` is currently supported. The contribution is
`((formulaValue - mean) / stdDev)^2`, with finite `mean` and positive finite
`stdDev`, both expressed in the space of the formula output. Non-finite results
produce an infinite penalty. Multiple constraints are summed independently;
correlations between external constraints are not supported.

Constraints read current physical parameter values without interpreting parameter
flags, changing priors, or changing parameter throws. They remain active for fixed,
free, or internally penalty-disabled parameters. Missing or ambiguous references
are rejected at initialization. Internal priors remain additive: avoid counting
the same measurement both as an internal prior and an external constraint.
`setCurrentParameterValuesAsPrior()` does not change external measurement means.

The Python interface exposes `getExternalConstraints()` on `LikelihoodInterface`,
then `getGaussianConstraintList()` and `eval()`. Each Gaussian
constraint exposes its configuration getters, `evalFormula()`,
`evalResidual()` and `eval()` for diagnostics.

Parameter references are resolved once during initialization. Evaluations read
the current values through cached const pointers; parameter storage must remain
alive and stable for the lifetime of the bound constraints.
