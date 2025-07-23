## parameterSetList

[< back to parent (propagatorConfig)](Propagator.md)

### Description

Parameter sets contain lists of parameters that are grouped together by the
fit, and which may share a prior covariance matrix.  Examples parameter
sets might be "oscillation parameters", "flux systematics", "cross section
systematics", and "detector systematics", but GUNDAM handles them all the
same, so you can use any name, and assign any meaning to the different
sets.  Parameter set lists occur in two contexts.  The parameters are
defined in the parametersManagerConfig which will have the syntax in the
YAML file of

```yaml
parametersManagerConfig:
    parameterSetList:
        - name: "First set"
          <and so on>
```

The other context is the through the `parameterInjection` which follows the
same syntax, and provides a way to explicitly set the value of the
parameters (primarily to override the values in the
parametersManagerConfig.

Each parameter set is defined using the fields in the next section.  The
`name` is the only required field in a parameter set, however at least one
parameter must be defined.  Parameters are defined by using the
`parameterDefinitions` which will contain a list of parameter definitions.
Parameters can also be defined using a prior covariance matrix, and vectors
of parameter names, prior values, and bounds (see `covarianceMatrix`,
`parameterNameList`, and `parameterPriorValueList`).

The event reweighting dials that depend on the parameters should be defined
in the `dialSetDefinitions`.  This defines all of the event reweighting
associated with this set of parameters.  For simple sets of parameters
(which only have one parameter per dial), the dial definitions can be
defined as part of the `parameterDefinitions`, but this practice is
discouraged.  If multi-dimensional dials are used, then a dialSetDefinition
is required.

If a parameter set has a prior covariance, it may be decomposed, and it's
eigen vectors can be used as the effective fit parameters.  This is
primarily needed with the prior covariance is singular (i.e. it has
extremely large correlations between parameters).  In this case, the
`eigenValueThreshold` may be set to ignore very small parameters.
Typically a singular covariance matrix will have one (or several) eigen
values that are approximately zero (e.g. with a magnitude of about 1E-14),
so an `eigenValueThreshold` of 1E-12 is usually appropriate.

### Parameter set config options

| Field                                | Type             | Description                                                  | Default |
|--------------------------------------|------------------|--------------------------------------------------------------|---------|
| name                                 | string           | Required: The parameter set name                             |         |
| [parameterDefinitions](Parameter.md) | list(Json)       | The list parameters that are part of this set                |         |
| [dialSetDefinitions](Dial.md)        | list(Json)       | The list of "dials" that modify the model                    |         |
| isEnabled                            | bool             | Is the set is enabled for the fit                            |         |
| isScanEnabled                        | bool             | Should the set be "scanned" by the parameter scanner         |         |
| nominalStepSize                      | double           | The step scale used by the minimizer                         | 1.0     |
| printDialSetSummary                  | bool             | Should the dial set summary be printed                       | false   |
| printParameterSummary                | bool             | Should the parameter set summary be printed                  | false   |
| parameterLimits                      | [double, double] | Define a range for all parameters in this set                |         |
| enablePca                            | bool             | Fix parameter if 1 sigma scan doesn't change LLH (DANGEROUS) | false   |
| enableThrowToyParameters             | bool             | This set should be thrown when generating toy data           | true    |
| releaseFixedParametersOnHesse        | bool             | Fixed parameters are considered for HESSIAN                  | false   |
| parameterDefinitionFilePath          | string           | Name of a file defining parameters and covariance            | empty   |
| covarianceMatrix                     | string           | Name of TH2D covariance in parameter definition file         | empty   |
| parameterNameList                    | string           | Name of TObjArray of strings for parameter names             | empty   |
| parameterPriorValueList              | string           | Name of TVectorD with prior values for each parameter        | empty   |
| parameterLowerBoundsList             | string           | Name of TVectorD with parameter lower bounds                 | empty   |
| parameterUpperBoundsList             | string           | Name of TVectorD with parameter lower bounds                 | empty   |
| enableOnlyParameters                 | list             | List of enabled parameter for this set (others disabled)     | empty   |
| disableParameters                    | list             | List of disabled parameters for this set (others enabled)    | empty   |
| useEigenDecompForThrows              | bool             | Throw toys in eigen decomposed basis                         | false   |
| enableEigenDecomp                    | bool             | Eigen decompose the prior covariance matrix                  | false   |
| allowEigenDecompWithBounds           | bool             | Allow bounds to be ignored when eigen decomposing            | false   |
| maxNbEigenParameters                 | int              | Only use this many eigen parameters                          | inf     |
| maxEigenFraction                     | double           | Fraction of total "eigen" power to use                       | 1.0     |
| eigenValueThreshold                  | double           | Ignore eigenvectors with eigenvalues below this threshold    | 0.0     |

### JSON sub-structures

#### parameterDefinitions

The `parameterDefinitions` field holds a list that defines the parameters that are part of this set.  The fields for each [parameter](Parameter.md) vary depending on the circumstances, but each definition must contain a parameter name.  For example
```yaml
parameterDefinitions:
    - parameterName "First Parameter"
      <other fields>

    - parameterName "Second Parameter"
      <other fields>
```

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

#### parameterLimits options

| Options  | Type   | Description                          | Default |
|----------|--------|--------------------------------------|---------|
| minValue | double | min value of parameters from the set | -inf    |
| maxValue | double | max value of parameters from the set | +inf    |
