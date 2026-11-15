## Parameter

[< back to parent (ParameterSet)](./ParameterSet.md)

### Description

Define a parameter of the fit.  The parameter should have a name,
but you can use the index instead (not encouraged).  Use `stdDev` to define an individual Gaussian constraint or override the
standard deviation from a prior covariance matrix. Overrides preserve the
correlations from the input matrix. Without a covariance matrix, parameters
with a `stdDev` are independent. An unset standard deviation defaults to
NaN and makes the parameter free (no penalty). `priorType` is deprecated.
For compatibility, `priorType: Flat` is converted to `stdDev: nan` with a warning, even when a finite standard deviation is provided.

Standard deviations must be positive and finite, or NaN for a free parameter.
For free parameters without a standard deviation, `parameterStepSize` (or the
set's `nominalStepSize`) must be explicitly defined, positive and finite.
It supplies the scale for minimization and scans; no default step is assumed.

### Config options

| Option                           | Type         | Description                                                | Default      |
|----------------------------------|--------------|------------------------------------------------------------|--------------|
| name                             | string       | define parameter with name (expected)                      |              |
| parameterIndex                   | int          | define parameter with index                                |              |
| priorType                        | string       | DEPRECATED: use stdDev (Flat converts to nan)               | unset     |
| releasePriorCorrelations         | bool         | Remove correlations with other parameters; keep stdDev constraint | false |
| stdDev                           | double       | Individual prior standard deviation                       | covariance diagonal, or nan |
| priorValue                       | double       | prior value (expected)                                     | nan          |
| isEnabled                        | bool         | use parameter in the propagator                            | true         |
| isPenaltyDisabled                | bool         | Fix value in fit, and strip from covariance                | false        |
| isFixed                          | bool         | Fix the parameter in the fit (e.g. set MINUIT FixVariable) | false        |
| isFrozen                         | bool         | DEPRECATED: use isFixed                                    | false        |
| isThrown                         | bool         | throw parameter when generating toy                        | false        |
| parameterStepSize                | double       | Expected scale of the variation                            | unset          |
| parameterLimits                  | pair(double) | minimum and maximum for the domain                         | [-inf, +inf] |
| physicalLimits                   | pair(double) | minimum and maximum physical range                         | [-inf, +inf] |
| throwLimits                      | pair(double) | minimum and maximum range to throw                         | [-inf, +inf] |
| mirrorRange                      | pair(double) | boundaries for internal mirroring                          | [-inf, +inf] |
| cyclicRange                      | pair(double) | boundaries of the cyclic interval                          | [-inf, +inf] |
| [dialSetDefinitions](Dial.md   ) | list(json)   | Only for single dimension dials                            |              |

#### dialSetDefinitions

The `dialSetDefinitions` field holds a list that the defines the dials which translate the parameters into event weights.  The fields for each [dial](Dial.md) vary depending on the dial type, but each dial must contain the `dialType`, and the parameter names (in `dialInputList`) that are used.
```yaml
dialSetDefinitions:
    - dialType: "Spline"
      dialInputList:
          - name: "First Parameter"
          - name: "Second Parameter"
      <other fields>
```

#### releasePriorCorrelations

Set `releasePriorCorrelations: true` to keep a parameter's Gaussian constraint
while removing its prior correlations with every other parameter. Its standard
deviation is inherited from the input covariance or overridden by `stdDev`.
Correlations between the remaining parameters are preserved. This applies to
the penalty, prior throws, and eigen decomposition. The default is `false`.
Without an input covariance, parameters are already independent. A free parameter
(`stdDev: "nan"`) still contributes no penalty regardless of this option.

```yaml
- parameterIndex: 0
  name: independentPrior
  releasePriorCorrelations: true
  # Optional: override the inherited std-dev while keeping the Gaussian constraint
  stdDev: 0.3
```
