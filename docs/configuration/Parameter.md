## Parameter

[< back to parent (ParameterSet)](./ParameterSet.md)

### Description

Define a parameter of the fit.  The parameter should have a name,
but you can use the index instead (not encouraged).  Notice that
parameters with a Gaussian prior must be associated with a
covariance matrix.  It is not currently possible to specifically
define parameters with a Gaussian constraint, but without a
covariance.

### Config options

| Option                           | Type         | Description                           | Default      |
|----------------------------------|--------------|---------------------------------------|--------------|
| parameterName                    | string       | define parameter with name (expected) |              |
| parameterIndex                   | int          | define parameter with index           |              |
| priorType                        | string       | penalty shape (Gaussian/Flat)         | Gaussian     |
| priorValue                       | double       | prior value (expected)                | nan          |
| isEnabled                        | bool         | use parameter in the propagator       | true         |
| isFixed                          | bool         | use parameter in the propagator       | false        |
| parameterStepSize                | double       | Expected scale of the variation       | 1.0          |
| parameterLimits                  | pair(double) | minimum and maximum allowed values    | [-inf, +inf] |
| throwLimits                      | pair(double) | minimum and maximum range to throw    | [-inf, +inf] |
| mirrorRange                      | pair(double) | boundaries for internal mirroring     | [-inf, +inf] |
| [dialSetDefinitions](Dial.md) | list(json)   | Only for single dimension dials       |              |

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
