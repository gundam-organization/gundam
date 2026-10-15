## Parameter

[< back to parent (ParameterSet)](./ParameterSet.md)

### Description

Define a parameter of the fit.  The parameter should have a name,
but you can use the index instead (not encouraged).  Use `stdDev` to define an individual Gaussian constraint or override the
standard deviation from a prior covariance matrix. Overrides preserve the
correlations from the input matrix. Without a covariance matrix, parameters
with a `stdDev` are independent. An unset standard deviation defaults to
NaN and makes the parameter free (no penalty). `priorType: Flat` also removes
the constraint, even when a standard deviation is provided.

Standard deviations must be positive and finite, or NaN for a free parameter.
For free parameters without a standard deviation, `parameterStepSize` (or the
set's `nominalStepSize`) supplies the scale for minimization and scans.

### Config options

| Option                           | Type         | Description                                                | Default      |
|----------------------------------|--------------|------------------------------------------------------------|--------------|
| name                             | string       | define parameter with name (expected)                      |              |
| parameterIndex                   | int          | define parameter with index                                |              |
| priorType                        | string       | penalty shape (Gaussian/Flat)                              | Gaussian     |
| stdDev                           | double       | Individual prior standard deviation                       | covariance diagonal, or nan |
| priorValue                       | double       | prior value (expected)                                     | nan          |
| isEnabled                        | bool         | use parameter in the propagator                            | true         |
| isPenaltyDisabled                | bool         | Fix value in fit, and strip from covariance                | false        |
| isFixed                          | bool         | Fix the parameter in the fit (e.g. set MINUIT FixVariable) | false        |
| isFrozen                         | bool         | DEPRECATED: use isFixed                                    | false        |
| isThrown                         | bool         | throw parameter when generating toy                        | false        |
| parameterStepSize                | double       | Expected scale of the variation                            | 1.0          |
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
