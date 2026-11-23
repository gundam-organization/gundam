# Parameters and Dials

## parameterSetList

[< back to parent (propagatorConfig)](configuration.md#propagator-configuration)

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
| [parameterDefinitions](#parameter) | list(Json)       | The list parameters that are part of this set                |         |
| [dialSetDefinitions](#dial-definition)        | list(Json)       | The list of "dials" that modify the model                    |         |
| isEnabled                            | bool             | Is the set is enabled for the fit                            |         |
| isScanEnabled                        | bool             | Should the set be "scanned" by the parameter scanner         |         |
| nominalStepSize                      | double           | Step scale; required for free parameters without parameterStepSize | nan     |
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
| enableOnlyParameters                 | list(Json)       | List of enabled parameters for this set (others disabled)    | empty   |
| disableParameters                    | list(Json)       | List of disabled parameters for this set (others enabled)    | empty   |
| useEigenDecompForThrows              | bool             | Throw toys in eigen decomposed basis                         | false   |
| enableEigenDecomp                    | bool             | Eigen decompose the prior covariance matrix                  | false   |
| allowEigenDecompWithBounds           | bool             | Allow bounds to be ignored when eigen decomposing            | false   |
| maxNbEigenParameters                 | int              | Only use this many eigen parameters                          | inf     |
| maxEigenFraction                     | double           | Fraction of total "eigen" power to use                       | 1.0     |
| eigenValueThreshold                  | double           | Ignore eigenvectors with eigenvalues below this threshold    | 0.0     |

### JSON sub-structures

#### parameterDefinitions

The `parameterDefinitions` field holds a list that defines the parameters that are part of this set.  The fields for each [parameter](#parameter) vary depending on the circumstances, but each definition must contain a parameter name.  For example
```yaml
parameterDefinitions:
    - parameterName "First Parameter"
      <other fields>

    - parameterName "Second Parameter"
      <other fields>
```

#### dialSetDefinitions

The `dialSetDefinitions` field holds a list that the defines the dials which translate the parameters into event weights.  The fields for each [dial](#dial-definition) vary depending on the dial type, but each dial must contain the `dialType`, and the parameter names (in `dialInputList`) that are used.
```yaml
dialSetDefinitions:
    - dialType: "Spline"
      dialInputList:
          - name: "First Parameter"
          - name: "Second Parameter"
      <other fields>
```

#### enableOnlyParameters and disableParameters options

The `enableOnlyParameters` and `disableParameters` fields hold lists that select parameters by name or by index.  `enableOnlyParameters` disables every parameter in the set except the selected ones.  `disableParameters` disables only the selected parameters, leaving the other parameters enabled.

| Options | Type   | Description                                           | Default |
|---------|--------|-------------------------------------------------------|---------|
| name    | string | Select a parameter using its name                     |         |
| index   | int    | Select a parameter using its zero-based index in set  |         |

For example:
```yaml
enableOnlyParameters:
    - name: "First Parameter"
    - index: 2

disableParameters:
    - name: "Second Parameter"
    - index: 3
```

#### parameterLimits options

| Options  | Type   | Description                          | Default |
|----------|--------|--------------------------------------|---------|
| minValue | double | min value of parameters from the set | -inf    |
| maxValue | double | max value of parameters from the set | +inf    |

### Independent parameter constraints

A covariance file is optional. For example, this set has two independent
Gaussian constraints and one free parameter:

```yaml
name: independentParameters
isEnabled: true
nominalStepSize: 0.1
parameterDefinitions:
  - name: normalization
    priorValue: 1.0
    stdDev: 0.2
  - name: offset
    priorValue: 0.0
    stdDev: 0.5
  - name: unconstrained
    priorValue: 1.0
    parameterStepSize: 0.1
```

The penalty is the sum of squared deviations divided by the respective
variances. The free parameter contributes no penalty. The same `stdDev`
field can override individual uncertainties when a covariance matrix is loaded;
the input correlations are preserved.

## Parameter

[< back to parent (ParameterSet)](#parametersetlist)

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
| [dialSetDefinitions](#dial-definition) | list(json)   | Only for single dimension dials                            |              |

#### dialSetDefinitions

The `dialSetDefinitions` field holds a list that the defines the dials which translate the parameters into event weights.  The fields for each [dial](#dial-definition) vary depending on the dial type, but each dial must contain the `dialType`, and the parameter names (in `dialInputList`) that are used.
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

## Dial Definition

[< back to parent (ParameterSet)](#parametersetlist)

[< back to parent (ParameterSet)](#parameter)

### Description

Describe an element in a list of dials.  A dial describes how parameter
values should be translated into a weight that will be used to multiply
events that are part of the model.  All of the weights must be positive,
and will generally be bounded on the high-side (see `minDialResponse` and
`maxDialResponse`). GUNDAM provides several types of weighting functions
(both single and multiple dimensions), as well as the ability to call
external libraries to calculate weights.

### Config options that apply to all dials.

| Option                 | Type         | Description                                                            | Default |
|------------------------|--------------|------------------------------------------------------------------------|---------|
| dialType               | string       | See below                                                              |         |
| options                | string or object | Type-specific settings: interpolation subtype or structured dial configuration | empty |
| isEnabled              | bool         | Dial is not used if this is false                                      | true    |
| applyOnDataSets        | list(string) | list (regex) of datasets the corresponding dials will apply to         | `["*"]` |
| applyCondition         | string       | Only apply this dial with formula evaluates to non-zero                | empty   |
| applyConditions        | list(json)   | Multiple condition formulas (only apply dial if all are true) formulas | n/a     |
| minDialResponse        | double       | Minimum weight value returned by the dial response (should be zero)    | -inf    |
| maxDialResponse        | double       | Maximum weight returned by the dial (typically you want about ten)     | inf     |
| binningFilePath        | string       | binning definition of a set of spline (along with `dialsFilePath`)     |         |
| useMirrorDial          | bool         | enable dial mirroring along the edges                                  | false   |
| mirrorLowEdge          | double       | low edge where mirroring applies                                       |         |
| mirrorHighEdge         | double       | upper edge where mirroring applies                                     |         |
| allowDialExtrapolation | bool         | Allow splines to be extrapolated past the first and last knots         | false   |
| printDialsSummary      | bool         | extra verbose                                                          | false   |
| definitionRange        | pair(double) |                                                                        |         |
| mirrorDefinitionRange  | pair(double) |                                                                        |         |
| dialInputList          | list(json)   | List of parameters for this dial                                       | empty   |
| dialsTreePath          | string       | ROOT tree path, optionally followed by `:branch` for binned dials       |         |
| dialsBranch            | string       | Explicit object branch in the binned dial tree                         |         |
| binnedDialsPath        | string       | Binned dial source as `file:tree:branch`                                |         |
| dialsBinningDefinition | list(object) | Explicit bin dimensions read from the dial TTree; exclusive with `binning` |       |

### Binned dials stored in a ROOT tree

Select the object branch explicitly using any of these equivalent forms:

```yaml
binnedDialsPath: "splines.root:EB_bin_C_nu:Graph"
binning: "binning.txt"
```

```yaml
dialsFilePath: "splines.root"
dialsTreePath: "EB_bin_C_nu:Graph"
binning: "binning.txt"
```

```yaml
dialsFilePath: "splines.root"
dialsTreePath: "EB_bin_C_nu"
dialsBranch: "Graph"
binning: "binning.txt"
```

The selected curve branch must contain `TGraph` (including subclasses) or
`TSpline3` objects. `dialType` still selects the interpolation method.

Choose exactly one bin source:

- `binning`: supply an external bin definition (file or inline configuration).
  Entry `i` of the TTree corresponds to bin `i`. Their counts must match.
- `dialsBinningDefinition`: create one bin per TTree entry using only the
  explicitly declared scalar branches. For example:

```yaml
dialsFilePath: "binnedDials.root"
dialsTreePath: "EB_bin_C_nu"
dialsBranch: "Graph"
dialsBinningDefinition:
  - {name: Enu, minValueBranch: minEnu, maxBranchValue: maxEnu}
  - {name: theta_mu, minValueBranch: minThetaMu, maxBranchValue: maxThetaMu}
  - {name: NeutrinoCode, valueBranch: NeutrinoCode}
```

`name` identifies the event variable. `minValueBranch` and `maxBranchValue`
supply a continuous interval `[min, max)`; both branches are required and
`min < max`. `valueBranch` supplies an equality selection. These two forms
cannot be mixed within a dimension. Numeric scalar branches (including integer,
float and double) are read as numbers, without forcing integer conversion.
Bounds may be infinite, but not NaN; equality values must be finite. Variable
names must be non-empty and unique within the definition.

`binning` and `dialsBinningDefinition` cannot be combined. The latter requires a
TTree source and cannot be used with `dialsList`.

Only declared branches participate in the bin selection. Other metadata,
including `kinematicBin`, is ignored. The former automatic interpretation of
scalar leaves as integer selections has been removed. Bins and curves retain
tree entry order; when a curve needs no dial, its bin is removed with it.

Redundant file, tree or curve-branch declarations must agree. For example,
`dialsTreePath: "EB_bin_C_nu"` may accompany the full `binnedDialsPath`.
Explicit tree branches cannot be combined with `dialsList`.

For backward compatibility, omitting the curve branch keeps the historical
choice: `Spline` for `dialType: Spline`, `Graph` for `dialType: Graph`. This
implicit curve-branch selection emits a deprecation warning. The existing
`dialsFilePath` plus `dialsList` object-array syntax is unchanged.

`options` is the common key for type-specific settings. `Spline`, `Graph` and
`Surface` retain their string options (for example, `options: Bilinear`).
`Formula`, `CompiledLibDial`, `Composite` and `ExternalWeight` use an object.
The old keys `dialSubType` and `dialConfig` remain readable aliases, with a
rename warning. If multiple aliases are supplied, their values must agree.

### Config options that apply to specific types of dials

| Option                 | Type         | Description                                                            | Default |
|------------------------|--------------|------------------------------------------------------------------------|---------|
| dialBranchData         | string or object | Read a ROOT object branch or a pair of TArray branches (see below). `dialLeafName` and `treeExpression` remain aliases. | |
| dialsFilePath          | string       | root file containing the set of dials                                  |         |
| dialsList              | string       | path within root file to the list of dials                             |         |
| tableConfig            | json         | Definition of functions to call for Tabular dials                      | empty   |
| knotResponseLimits     | range        | Clamp knot responses before building Graph and Spline dials (see below) | unbounded |

`knotResponseLimits` bounds the knot ordinates when constructing a Graph or
Spline dial, before constant and two-point optimizations. Use `[0.5, 2.0]`
for both bounds, `{min: 0.5}` for a lower bound only, or `{max: 2.0}` for an
upper bound only. For example:

```yaml
dialType: Spline
dialBranchData: graph
knotResponseLimits: {max: 2.0}
```

Spline slopes are recalculated if knots change. Interpolation and extrapolation
can still exceed these bounds between or beyond knots; `minDialResponse` and
`maxDialResponse` bound the evaluated response. Omitting `knotResponseLimits`
leaves the input knots unchanged.

### Configuration for different dial types

#### `dialType: Composite`

A composite evaluates a directed acyclic graph of ordinary dials and exposes
one output. Only this output multiplies the event weight or updates the
observable selected by `applyToVariable` / `variableResponseMode`.

```yaml
dialSetDefinitions:
  - name: combinedResponse
    dialType: Composite
    options:
      nodes:
        - name: first
          dialType: Formula
          dialInputList: [{name: value, parameterName: par0}]
          options:
            formulaStr: "[value] * [value]"
        - name: second
          dialType: Formula
          dialInputList: [{name: value, parameterName: par1}]
          options:
            formulaStr: "exp([value])"
        - name: combine
          dialType: Formula
          dialInputList:
            - {name: x, dialResponse: first}
            - {name: y, dialResponse: second}
          options:
            formulaStr: "[x] + [y]"
      output: combine
```

`parameterName` references a fit parameter in the owning parameter set;
`parameterIndex` references its zero-based index in that set. `dialResponse`
names another node in the same graph. Formula nodes also accept `eventVariable`,
naming a loaded event variable. Exactly one of these four keys is required
for each node input. Inputs may mix these sources. `inputIndex` is unsupported.
Formula inputs require a unique `name`, used as `[name]` in the expression.
Other dial types retain positional argument order. Node declaration order
does not matter; shared dependencies are evaluated once per graph evaluation.
Node names must be non-empty and unique, dependencies must exist, and every
node must contribute to `output`. Cycles are rejected.

Omit `dialInputList` on Composite definitions, including nested composites.
Their parameter dependencies are inferred from all their nodes, so a change
in any referenced fit parameter invalidates the final response cache. This
also applies to composites attached to a single parameter. Other nodes need
an explicit, non-empty `dialInputList`.

Nodes use the ordinary dial configuration, including interpolation options,
response limits, extrapolation and input mirroring. Intermediate
responses are unbounded by default so signed quantities can pass between
nodes. Explicit node limits are applied before downstream evaluation. The
composite's final response retains the usual weight/observable response limits.

| Node type | Supported inputs / loading |
|-----------|----------------------------|
| `Normalization` | One input of either source kind |
| `Formula` | Multiple parameter/response inputs, plus `eventVariable` inputs |
| `CompiledLibDial` | Multiple parameter/response inputs |
| `Graph`, `Spline` | One input; ordinary binned-file or event-branch loading, including paired TArray branches |
| `Surface` | Two inputs; ordinary binned-file or event-branch loading |
| `Composite` | Another graph; its parameter dependencies are inferred recursively |
| `Tabulated`, `Kriged`, `ExternalWeight` | Parameter inputs only; their prepared tables/buffers can feed downstream nodes |

Prepared factories cannot consume `dialResponse` inputs, including through a
nested composite, or inputs mirrored by an enclosing composite. Their updates
run before per-event graph evaluation. These cases are rejected explicitly.
`CompiledLibDial` exposes no arity metadata: its implementation must agree
with the configured input list.

Put the final `applyToVariable` and `variableResponseMode` on the composite.
Put `binning` and `dialBranchData` on individual nodes. Node dataset filters,
conditions and bin selections use the normal loading rules and select a dial
for the **same event**. If the composite applies but a required node has no
dial (including an empty branch curve), cache construction fails. Constant
unit responses are preserved as valid node values. Disabled nodes and the
indirect selectors `dialsDefinitions` / `parametersBinningPath` are unsupported
inside a composite; use direct node definitions with explicit inputs.

This first implementation uses the standard propagation engine, with its
normal multithreaded event evaluation and response caching. `CacheManager`
automatically disables itself when composites are present.

#### `dialType: Normalization`

| Option                 | Type         | Description                                                            | Default |
|------------------------|--------------|------------------------------------------------------------------------|---------|
| parametersBinningPath  | string       | create a dedicated norm dial according to the parameter binning        |         |

Treat the parameter as a simple weight.  In this case, the parameter must be bounded to always be positive.

#### `dialType: Spline`

| Option                 | Type         | Description                                                            | Default |
|------------------------|--------------|------------------------------------------------------------------------|---------|
| dialBranchData         | string or object | Read a ROOT object branch or a pair of TArray branches (see below). `dialLeafName` and `treeExpression` remain aliases. | |
| dialsFilePath          | string       | root file containing the set of dials                                  |         |
| dialsList              | string       | path within root file to the list of dials                             |         |

Translate a parameter value to a weight using an event-by-event spline (A
spline is a piece-wise continuous cubic function with continuous first and
second derivatives).  Each event for which this dial is applicable must
have a spline defined in the input file.  The spline will be contained in a
leaf named by the `dialLeaf`field, and must contain a TGraph or a TSpline3 object.
The `Spline` dial type can have several `dialSubType` values

- not-a-knot : Calculate the slopes using the "not-a-knot" criteria.  This
  is the default criteria used by the ROOT TSpline3 class. This guarantees
  that the third derivative at the first two and last two points is zero.
- natural : Calculate the slopes applying the "natural" criteria. This
  guarantees that the second derivatives at the end points will be zero.
- catmull-rom : Estimate the slopes using the Catmull-Rom criteria.  This
  uses the average slopes for the points before and after (also known as
  the "pixar" spline based on its use in animation.
- akima : Estimate the slopes using the Akima criteria.
- monotonic : Apply the monotonic critera to the slopes before they are
  used.  This applies to "not-a-knot", "natural" "catmull-rom", and "akima"
  splines.  In each case, the slopes are first estimated using the defined
  algorithm, and then the Fritsche-Carlson criteria is used to adjust the
  slopes so that the function is monotonic at every point, except for cusps
  where the slope will be zero.
- uniformity(tolerance) : A spline is considered to have uniform point
  spacing when the point spacing stays within this tolerance.  For example,
  `uniformity(1E-3)` means that the difference between the point to point
  spacings must stay below `1E-3`.

#### `dialType: Graph`

| Option                 | Type         | Description                                                            | Default |
|------------------------|--------------|------------------------------------------------------------------------|---------|
| dialBranchData         | string or object | Read a ROOT object branch or a pair of TArray branches (see below). `dialLeafName` and `treeExpression` remain aliases. | |
| dialsFilePath          | string       | root file containing the set of dials                                  |         |
| dialsList              | string       | path within root file to the list of dials                             |         |

Translate a parameter value to a weight piece-wise linear function.  Be
very careful when using a Graph dial since graphs will usually have
discontinuous derivatives, and cannot be reliably used in a minimizer.  A
minimizer requires that the function value be continuous, and have
continuous first and second derivatives. The graph will be contained in a
leaf named by the `dialLeaf` field, and must contain a TGraph object.

#### `dialType: Surface`

| Option                 | Type         | Description                                                            | Default |
|------------------------|--------------|------------------------------------------------------------------------|---------|
| dialBranchData         | string or object | Read a ROOT object branch or a pair of TArray branches (see below). `dialLeafName` and `treeExpression` remain aliases. | |
| dialsFilePath          | string       | root file containing the set of dials                                  |         |
| dialsList              | string       | path within root file to the list of dials                             |         |

Translate two parameter values into a surface.  The surface is defined in a
TH2 which is contained in a leaf named in the `dialLeaf`.  A surface dial
can have the following sub type fields

- Bilinear : Use bilinear interpolation between the points defined in the
  TH2.  The TH2 must define a regular grid of knots.
- Bicubic : Use bicubic interpolation between the points defined in the TH2
  The TH2 must define a regular grid of knots, and the bicubic
  interpolation is based on the catmull-rom definition.

#### `dialType: Tabulated`

| Option                 | Type         | Description                                                            | Default |
|------------------------|--------------|------------------------------------------------------------------------|---------|
| tableConfig            | json         | Definition of functions to call for Tabulate dials                      | empty   |

Use a precalculated table of weights.  The table may be refilled for each
iteration of the fitter. The method of filling the table is defined using
the [tableConfig](#tabulated-dials) option.

#### `dialType: Kriged`

| Option                 | Type         | Description                                                            | Default |
|------------------------|--------------|------------------------------------------------------------------------|---------|
| tableConfig            | json         | Definition of functions to call for Kriged dials                      | empty   |

Use a precalculated table of weights.  The table may be refilled for each
iteration of the fitter. The method of filling the table is defined using
the [tableConfig](#kriged-dials) option.

#### `dialType: ExternalWeight`

| Option            | Type      | Description                                             | Default |
|-------------------|-----------|---------------------------------------------------------|---------|
| type              | string    | Worker implementation; currently `PythonWorker`         |         |
| inputEventVarList | list(str) | Optional event variables exposed to the worker          | empty   |
| workerConfig      | json      | Configuration specific to the selected worker           |         |

Use an external worker to calculate one weight per selected event.
`PythonWorker` is the current implementation; other worker types can use the
same shared-memory inputs, parameters, and output weights. JSON is only used
as a lightweight control protocol. See [ExternalWeightDials](#external-weight-dials).

#### `dialType: Formula`

Apply a weight using a ROOT formula.  This is not fully supported since it
cannot be efficiently applied with GPU acceleration.

Formula inputs can mix fit parameters with loaded event variables:

```yaml
dialType: Formula
dialInputList:
  - {name: delta, parameterName: delta}
  - {name: alpha, parameterName: alpha}
  - {name: q3, eventVariable: q3}
options:
  formulaStr: "[delta] + [alpha] * (0.056 * [q3] - 41.6)"
minDialResponse: -40.
maxDialResponse: 40.
```

Each input has a unique, non-empty `name`, which is the local alias used as
`[name]` in the expression. It must select exactly one source:

- `parameterName`: the fit parameter name in the supervised parameter set;
- `parameterIndex`: its zero-based index in that set;
- `eventVariable`: a loaded event variable, including `variableDict` aliases.

Explicit `parameterName` and `parameterIndex` references are also accepted by
other dial types. For these dials, the order in `dialInputList` determines the
argument order; `name`, when provided, does not select the source parameter.

For example, `{name: offset, parameterName: delta}` binds `[offset]` to `delta`.
Input declaration order does not determine ROOT parameter order: names are
resolved once when configuring the dial. Positional `x[i]` expressions and
parameter references using only `{name: delta}` are unsupported for Formula.

Inside a composite, use the same `{name: delta, parameterName: delta}` or
`{name: alpha, parameterIndex: 1}` syntax for fit parameter inputs. Event references remain
`{name: q3, eventVariable: q3}`; responses from other nodes use
`{name: shift, dialResponse: nodeName}`.

Event variables are requested automatically. Bracketed source names such as
`{name: momentum, eventVariable: "[q3]"}` are accepted. Different input aliases
may reference the same event variable and share its captured value. Event-only
and binned formulas are supported. Binned formulas retain constants named
after their binning axes, for example `[X]` for the center of the selected X
bin. A declared input with the same name overrides that constant. Undeclared
references are rejected instead of silently evaluating as zero.

Evaluation supplies a call-local parameter array to `TFormula::EvalPar`,
without modifying the shared TFormula parameters. When declaration order
already matches ROOT's parameter order and there are no bin constants, the
existing input buffer is passed directly.

The values are captured for each event when building the dial cache, after
sorting events. They remain fixed until the cache is rebuilt. Parameter
changes trigger reevaluation through the normal input buffer. The source
`TFormula` and parameter buffer remain shared across events; each event owns
its captured values and response cache. Asimov copies retain their responses.

`eventVariable` inputs are supported only for `Formula` dials. `CacheManager`
automatically disables itself for event formulas, using the original engine.

##### Example: bin constants

`constantNames_` is an internal evaluator argument, not a YAML field. For a
binned Formula dial, GUNDAM fills it from the names of the binning axes and
sets their values to the centers of the selected bin:

```yaml
dialSetDefinitions:
  - name: energyBinResponse
    dialType: Formula
    dialInputList:
      - {name: scale, parameterName: scale}
    options:
      formulaStr: "[scale] * [Enu]"
    binning:
      binningDefinition:
        - name: Enu
          edges: [0., 1., 2., 4.]
```

Here `[scale]` is supplied from the fit parameter. `[Enu]` is a bin constant:
an event with `Enu` in `[1, 2)` uses `[Enu] = 1.5`. The supervised parameter
set must define `scale`, and the dataset must provide `Enu` for bin selection.

### applyConditions options

| Option                            | Type               | Description                                                      | Default |
|-----------------------------------|--------------------|------------------------------------------------------------------|---------|
| exp / expression / var / variable | string             | formula condition that applies on every dial of the set          |         |
| allowedRanges                     | list(pair(double)) | list of ranges (min, max) where the dials of the set apply       |         |
| excludedRanges                    | list(pair(double)) | list of ranges (min, max) where the dials of the set don't apply |         |
| allowedValues                     | list(double)       | list of values where the dials of the set apply                  |         |
| excludedValues                    | list(double)       | list of values where the dials of the set apply                  |         |

### Event-by-event branch data

For a `Spline` or `Graph` dial, a pair of TArray branches can supply the
parameter values (X) and responses (Y) for each event:

```yaml
dialType: Spline
dialBranchData:
  parameterValuesTArray: spline_parameter_values
  weightValuesTArray: spline_responses
```

Both names refer to branches of the input TTree. The branches must contain
TArray objects, for example TArrayF or TArrayD; the two arrays may use different
numeric types but must have the same length for each entry. Points are paired
by array index and processed using the same spline options, sorting, definition
range and extrapolation settings as TGraph inputs. Empty arrays produce no dial;
constant responses retain the existing constant-dial behavior.

Existing ROOT object inputs use a string:

```yaml
dialType: Spline
dialBranchData: spline_graph
```

Existing configurations using `dialLeafName: spline_graph` or
`treeExpression: spline_graph` remain supported. The array-pair form is limited
to `Spline` and `Graph` dials; it does not use the TClonesArray dial index.

### Applying a dial to an event variable

Any dial can target a stored event variable instead of the event weight:

```yaml
dialType: Normalization
applyToVariable: recoEnergy
variableResponseMode: Multiply
```

`variableResponseMode` accepts `Multiply` (the default), `Add`, or `Override`.
For each event and target variable, responses are composed as:

```text
value = (Override response, or loaded value) * product(Multiply responses) + sum(Add responses)
```

Multiple `Override` dials targeting the same variable of the same event are an
error. Disjoint application conditions may select different override dials.
Every propagation starts from the loaded reference value, so shifts do not
accumulate. Target variables hold double precision responses; other variables
retain their stored types. The reference is captured when the dial cache is built,
before the first propagation.

The loader stores target variables automatically and retains out-of-bin events
in samples binned with those variables by enabling `useDynamicBinning`. Initial
loading assigns bins using the loaded values. Propagation updates variables,
rebuilds bin contributions, computes event weights, and fills histograms.
Smoothing is enabled only through an explicit `binningResponse` configuration;
its fractions are recomputed using the updated variables.

`Add` and `Override` allow negative responses by default. Explicit
`minDialResponse` and `maxDialResponse` limits still apply. Variable responses
are excluded from the event weight and its global reweight cap. Sample selections
and event-to-dial associations remain those established at loading; derived
variables are not automatically recalculated after a target variable changes.
This feature currently requires the CPU propagation path. Leave CacheManager
disabled (`--cache-manager off` with `gundamFitter`); its support is outside this
implementation.

## Tabulated Dials

A tabulated dial provides an efficient way to do an event-by-event reweight
of an event based on a single truth variable and fitting parameters.  It is
an efficient way to handle a one-dimensional lookup table. For instance,
this can be used for long and short baseline oscillation fits that are
dependent on the true neutrino energy.  The table will be refilled for each
set of oscillation parameters.  An external library is used declare a look
up table, to fill the table for each iteration, and look up the position of
each event in the table based on truth variables.

The yaml for a Tabulated Dial is

```YAML
dialSetDefinitions:
  - dialType: Tabulated
    printDialsSummary: true
    dialInputList:
      - name: <fit-parameter1-name>
      - name: <fit-parameter2-name>
    tableConfig:
        - name: <table-name>                A table name passed to the
                                               library.
          bins:  <number>                   The (optional) expected number
                                               of bins in the table. Will
                                               be overridden by the
                                               initFunction.
          libraryPath: <path-to-library>    Location of the library
          initFunction: <init-func-name>    Initialization function name
          initArguments: [<arg1>, ...]      List of argument strings (e.g.
                                               input file names)
          updateFunction: <update-func-name> Function called to update table
          binningFunction: <bin-func-name>  Function to find bin index
          binningVariables: [<var1>, <var2>, ... ] Variables used for binning
                                               the table "X" coordinates
```

### Function to initialize the library

If `initFunction` is provided it will be called before calling the update or the binning functions.  It is called with the signature:

```C++
    extern "C"
    int initFunc(const char* name,
                 int argc, const char* argv[],
                 int bins);
```
* name -- The name of the table
* argc -- number of arguments
* argv -- argument strings.  The arguments are defined by the library, but are usually things like input file names for the lookup table information.
* bins -- The suggested size the table.  The library must choose an appropriate binning that will be used for the table.

The return value must be less than or equal to zero for failure, and otherwise, the provide the number of elements needed to store the table (usually, the same as the input value of "bins").

### Library function to update the table

The update function is called before the events are reweighted.  The `updateFunction` signature is:

```C++
    extern "C"
    int updateFunc(const char* name,
                   double table[], int bins,
                   const double par[], int npar);
```
* name  -- table name
* table -- address of the table to update
* bins  -- The size of the table
* par   -- The parameters.  Must match parameters define in the dial definition
* npar  -- number of parameters

The function must return 0 for success, and any other value for failure

The table will be filled with "bins" values calculated with uniform spacing between "low" and "high".  If bins is one, there must be one value calculated for "low", if bins is two or more, then the first point is located at "low", and the last point is located at "high".  The step between the bins is (high-low)/(bins-1).  Examples:
```
    bins = 2, low = 1.0, high = 6.0
       values calculated at 1.0 and 6.0
```
```
    bins = 3, low = 1.0, high = 6.0
       values calculated at 1.0, 3.5, and 6.0
```
While the update can (in principle) be done directly on the GPU, and have the table directly accessed by the Cache::Manager, the results must be copied back to the CPU so that the Propagator event weight calculation can be done.  The current implementation does not expose the methods to hand the table directly from the external library to the Cache::Manager kernels.  The calculation can also be done solely on the CPU, and the results will be copied to the GPU.

### Library function to index events

The binning function is called as the MC events are read into the internal structures, and return an index into the lookup table for an event.  The `binningFunction` signature is:
```C++
    extern "C"
    double binFunc(const char* name,
                   int nvar, const double varv[],
                   int bins);
```
* name -- table name
* nvar -- number of (truth) variables used to find bin
* varv -- array of (truth) variables used to find bin
* bins -- The number of bins in the table.

The function should return a double giving the fractional bin number greater or equal to zero and LESS THAN the maximum number of bins.  The  integer part determines the index of the value below the value to be interpolated, and the fractional part determines the interpolation between  the indexed bin, and the next.  This determines where to find the entry for  the input (truth) variables.

### Compiling the library

 The library should loadable by dlopen, so it needs be compiled with (at least)

`gcc -fPIC -rdynamic --shared -o <LibraryName>.so <source>`

## Kriged Dials

A kriged dial provides an efficient way to do an event-by-event reweight of an event based on fitting parameters when the lookup is indexed by more than one truth variable. For instance, this can be used for atmospheric neutrino oscillation probability tables where the event probability depends on both the neutrino direction, and the neutrino energy. The kriged dial uses a lookup table that must be filled when the fitting parameter values are changed. The table is managed by an external library that is used to declare the look up table, fill the table for each iteration, and look up the "kriging weights" in the table for each set of truth variables.

The yaml for a Kriged dial is

```YAML
dialSetDefinitions:
  - dialType: Kriged
    printDialsSummary: true
    dialInputList:
      - name: <fit-parameter1-name>
      - name: <fit-parameter2-name>
    tableConfig:
        - name: <table-name>                A table name passed to the
                                               library.
          bins:  <number>                   The (optional) expected number
                                               of bins in the table. Will
                                               be overridden by the
                                               initFunction.
          libraryPath: <path-to-library>    Location of the library
          initFunction: <init-func-name>    Initialization function name
          initArguments: [<arg1>, ...]      List of argument strings (e.g.
                                               input file names)
          updateFunction: <update-func-name> Function called to update table
          weightFunction: <bin-func-name>  Function to find bin index
          weightVariables: [<var1>, <var2>, ... ] Variables used to fill
                                               the arrays of table indices
                                               and constants used to calculate
                                               the event weight
```

#### Function to initialize the library

If `initFunction` is provided it will be called before calling the update or the binning functions.  It is called with the signature:

```C++
    extern "C"
    int initFunc(const char* name,
                 int argc, const char* argv[],
                 int bins);
```
* name -- The name of the table
* argc -- number of arguments
* argv -- argument strings.  The arguments are defined by the library, but are usually things like input file names for the lookup table information.
* bins -- The suggested size the table.  The library must choose an appropriate binning that will be used for the table.

The function must return less than or equal to zero for failure, and otherwise, the number of elements needed to store the table (often, the same as the input value of "bins").

#### Library function to update the table

The update function is called before the events are reweighted.  The `updateFunction` signature is:

```C++
    extern "C"
    int updateFunc(const char* name,
                   double table[], int bins,
                   const double par[], int npar);
```
* name  -- table name
* table -- address of the table to update
* bins  -- The size of the table
* par   -- The parameters.  Must match parameters define in the dial definition
* npar  -- number of parameters

The function should return 0 for success, and any other value for failure

While the update can (in principle) be done directly on the GPU, and have the table directly accessed by the Cache::Manager, the results must be copied back to the CPU so that the Propagator event weight calculation can be done.  The current implementation does not expose the methods to hand the table directly from the external library to the Cache::Manager kernels.  The calculation can also be done solely on the CPU, and the results will be copied to the GPU.

#### Library function to look up the events

The weight function is called as the MC events are read into the internal structures, and returns the indices and weights to use for an event.  The `weightFunction` signature is:

```C++
    extern "C"
    int weightFunc(const char* name, int bins,
                   int nvar, const double varv[],
                   int maxEntries,
                   int index[], double weights[]);
```
* name -- table name
* bins -- number of entries in the look up table
* nvar -- number of (truth) variables used to find bin
* varv -- array of (truth) variables used to find bin
* maxEntries -- The space allocated for index and weights
* index[] -- (output) an array filled with the index values used to weight the event.  It must hold at least `maxEntries` values
* weights[] -- (output) an array filled with the weights to be applied the table entries referenced in the index array.  It must hold at least `maxeEntries` values.

The function should return an int giving the number of entries that were filled in the index array.  The index and weights arrays are "parallel", so the first element of the index array is the index in the table that the weight in the first element of the weight array will be applied to.  The pseudocode for the event weight calculation is

```C++
double weight = 0;
for (int i=0; i<entries; ++i) weight += weights[i]*table[index[i]];
```

#### Compiling the library

The library should loadable by dlopen, so it needs be compiled with (at least)

`gcc -fPIC -rdynamic --shared -o <LibraryName>.so <source>`

#### An abbreviated theory of Kriging

Kriging provides a general, relatively efficient, implementation of a
multi-dimensional look up table for a function, when the function can be
calculated at a set of "sampled" points, and needs to be extrapolated to
"unsampled" points. The function value for the unsampled points is calculated
as the weighted average of selected sampled points (the weights and sampled
points can be pre-calculated).  Kriging is a special case of gaussian process
regression using a linear kernel, and was developed by D. Krige (Krige, D. G.
1951. "A statistical approach to some mine valuation and allied problems on the
Witwatersrand", Master's thesis, University of the Witwatersrand).

## External Weight Dials

[< back to parent (Dial)](#dial-definition)

External weight dials let GUNDAM delegate the evaluation of event weights to an external worker. `PythonWorker` is the first implementation; the common worker interface is also intended for precompiled external libraries.

This is useful when the response depends on many events at once, for example an oscillation engine that can evaluate all event weights in a vectorized CPU/GPU backend.

The external worker is persistent. GUNDAM loads the requested event inputs once, then each likelihood propagation only sends the current fit parameters and asks the worker to update the event weights.

### YAML Configuration

The dial is configured in a `dialSetDefinitions` entry:

```yaml
parameterSetList:
  - name: "OscillationParameters"
    isEnabled: true
    nominalStepSize: 0.1

    dialSetDefinitions:
      - dialType: ExternalWeight
        applyOnDataSets: [ "MySample" ]
        dialInputList:
          - name: "deltaMsq"
          - name: "sinSqTheta"
        options:
          type: "PythonWorker"
          useBinnedWeights: false
          inputEventVarList:
            - "[Enu]"
            - "[FlavorEmit]"
            - "[FlavorDetect]"
          workerConfig:
            pythonExecutable: "/path/to/venv/bin/python"
            evalScript: "./config/externalWeightWorker.py"
            scriptArgs: [ "--model-config", "./config/osc.yaml" ]

    parameterDefinitions:
      - name: "deltaMsq"
        priorValue: 2.5e-3
        priorType: Flat
        parameterLimits: [0.0, 0.01]
      - name: "sinSqTheta"
        priorValue: 0.5
        priorType: Flat
        parameterLimits: [0.0, 1.0]
```

#### Options

| Option                        | Type       | Description                                                     | Default |
|-------------------------------|------------|-----------------------------------------------------------------|---------|
| dialType                      | string     | Must be `ExternalWeight`                                        |         |
| dialInputList                 | list(json) | Fit parameters passed to the worker                             |         |
| options.type                  | string     | Worker implementation; currently `PythonWorker`                 |         |
| options.inputEventVarList     | list(str)  | Event variables made available to every worker                  | empty   |
| options.useBinnedWeights      | bool       | Evaluate one weight per configured bin instead of per event      | false   |
| options.binning                | json       | `BinSet` configuration used when `useBinnedWeights` is enabled  |         |
| options.workerConfig          | json       | Configuration specific to the selected worker                   |         |
| workerConfig.pythonExecutable | string     | Python executable used to start `PythonWorker`                  |         |
| workerConfig.evalScript       | string     | Python worker script                                            |         |
| workerConfig.scriptArgs       | list(str)  | Extra command-line arguments passed to the Python worker script | empty   |
| workerConfig.initScript       | string     | Currently ignored by the shared-memory Python worker            |         |

`inputEventVarList` entries use the same variable naming as other event formulas. In practice, prefer bracket notation for variables from the dataset `variableDict`:

```yaml
options:
  inputEventVarList: [ "[Enu]", "[FlavorEmit]", "[FlavorDetect]" ]
```

`inputEventVarList` is optional. Omit it when the weight depends only on fit parameters. Each configured input is evaluated while loading the dataset and stored in memory by the common worker layer.

When `useBinnedWeights` is enabled, `options.binning` uses the standard GUNDAM
`BinSet` syntax. GUNDAM creates one dispatcher per bin and `EventDialCache`
associates each event with the matching dispatcher. The Python worker receives
one input value per bin, evaluated at the bin center, and writes one weight
per bin. Binning variables are automatically added to `inputEventVarList`.
Every input variable requested by the worker must be defined in every bin.
Events outside the configured bins do not receive the external dial.

For example:

```yaml
options:
  type: PythonWorker
  useBinnedWeights: true
  binning:
    binningDefinition:
      - name: Enu
        nBins: 100
        min: 0.0
        max: 10.0
      - name: FlavorEmit
        values: [12, 14]
      - name: FlavorDetect
        values: [12, 14]
  workerConfig:
    pythonExecutable: "/path/to/venv/bin/python"
    evalScript: "./config/externalWeightWorker.py"
```

In binned mode, `command_["nBins"]` is the size of the NumPy arrays exposed
through `command_["inputs"]` and `command_["weights"]`.

At the moment, the shared-memory transport exposes event inputs as `float64` arrays. If an input is semantically an integer, such as a flavor code, the worker should cast it explicitly before using it.

### Python Worker Contract

The worker is launched as:

```bash
/path/to/python externalWeightWorker.py --worker
```

If `scriptArgs` is provided, GUNDAM inserts those arguments before `--worker`. For example:

```bash
/path/to/python externalWeightWorker.py --model-config ./config/osc.yaml --worker
```

Before starting the persistent worker, GUNDAM runs a light preflight check with:

```bash
/path/to/python -m py_compile externalWeightWorker.py
```

This does not accelerate the fit in a meaningful way, but it catches Python syntax errors early and fails before the likelihood initialization continues.

GUNDAM communicates with the worker through small JSON commands on stdin/stdout. The numerical arrays are not transferred through JSON. They are exposed as POSIX shared-memory buffers.

The worker must:

- read one JSON command per line from stdin,
- write exactly one JSON response per command to stdout,
- reserve stdout for JSON responses only,
- write logs/debug messages to stderr if needed,
- return `{"status": "ok"}` for successful commands.

#### Commands

`initialize` is sent once, before the first evaluation. It contains:

```json
{
  "command": "initialize",
  "nEvents": 10,
  "inputs": {
    "Enu": {
      "shmName": "gdmEW_12345_i0",
      "dtype": "float64",
      "shape": [10]
    },
    "FlavorEmit": {
      "shmName": "gdmEW_12345_i1",
      "dtype": "float64",
      "shape": [10]
    },
    "FlavorDetect": {
      "shmName": "gdmEW_12345_i2",
      "dtype": "float64",
      "shape": [10]
    }
  },
  "parameters": [
    { "name": "deltaMsq", "title": "#0_deltaMsq", "index": 0 },
    { "name": "sinSqTheta", "title": "#1_sinSqTheta", "index": 1 }
  ],
  "parameterBuffer": {
    "shmName": "gdmEW_12345_parameters",
    "dtype": "float64",
    "shape": [2]
  },
  "weights": {
    "shmName": "gdmEW_12345_w",
    "dtype": "float64",
    "shape": [10]
  }
}
```

`evaluate` is sent at each propagation where the fit parameters changed:

```json
{ "command": "evaluate" }
```

The worker should read the current parameter values from `parameterBuffer`, compute one weight per event, write the result into `weights`, then respond:

```json
{ "status": "ok" }
```

`shutdown` is sent when GUNDAM destroys the worker:

```json
{ "command": "shutdown" }
```

### Minimal Worker Example

GUNDAM preloads the worker protocol, including JSON parsing, shared-memory
attachment and response handling. A new-style script only needs a `run`
function. On every call, `command_` contains:

- `command_['command']`: `initialize`, `evaluate` or `shutdown`;
- `command_['inputs']`: a dictionary of NumPy arrays;
- `command_['parameters']`: a dictionary mapping parameter names to values;
- `command_['weights']`: the writable NumPy output array.

The `initialize` command also contains `parameterInfo`, with the original
parameter metadata. The function may return a response dictionary; when it
returns `None`, GUNDAM sends `{"status": "ok"}`. An optional `configure`
function receives `scriptArgs` before the first command, allowing the script
to parse its command-line arguments.

```python
import sys

import numpy as np


def configure(arguments_):
    # arguments_ contains workerConfig.scriptArgs. For example, with:
    #   scriptArgs: ["--baseline-km", "295"]
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-km", type=float, default=295.0)
    configure.options = parser.parse_args(arguments_)


def run(command_):
    command_name = command_["command"]

    if command_name == "initialize":
        # Inputs are already attached NumPy arrays at this point. The
        # parameterInfo list contains name/title/index metadata.
        print("Loaded parameters:", command_["parameterInfo"], file=sys.stderr)
        return {"status": "ok"}

    if command_name == "evaluate":
        parameters = command_["parameters"]
        enu = command_["inputs"]["Enu"]
        flavor_emit = np.rint(command_["inputs"]["FlavorEmit"]).astype(np.int64)
        flavor_detect = np.rint(command_["inputs"]["FlavorDetect"]).astype(np.int64)

        phase = 1.267 * parameters["deltaMsq"] * configure.options.baseline_km / enu
        transition_prob = parameters["sinSqTheta"] * np.sin(phase) ** 2
        command_["weights"][:] = np.where(
            flavor_emit == flavor_detect,
            1.0 - transition_prob,
            transition_prob,
        )
        return {"status": "ok"}

    if command_name == "shutdown":
        # GUNDAM closes the shared-memory handles after run() returns.
        return {"status": "ok"}

    return {"status": "error", "message": "unknown command: " + command_name}
```

`attach_array`, `respond`, `run_worker` and the `__main__` block are no
longer needed. GUNDAM calls `configure` once, then calls `run` for each
command and handles the shared-memory cleanup after `shutdown`.

### Internal Engine Flow

In the default mode, GUNDAM creates one `ExternalWeightDispatcher` per
selected event. In `useBinnedWeights` mode, it creates one dispatcher per
configured bin instead; `EventDialCache` associates each selected event with
the corresponding bin dispatcher.

After all events are loaded:

1. GUNDAM allocates one shared-memory `double` array per requested input variable (none when `inputEventVarList` is omitted).
2. GUNDAM fills those input arrays once from the selected events, or with bin-center values in binned mode.
3. GUNDAM allocates one shared-memory `double` array for the current fit parameters.
4. GUNDAM allocates one shared-memory `double` array for the output weights. Its size is `nEvents` in the default mode and `nBins` in binned mode.
5. GUNDAM starts the Python worker and sends the shared-memory names and array metadata in the `initialize` JSON command.

At each propagation:

1. GUNDAM writes the current fit parameter values into `parameterBuffer`.
2. GUNDAM sends the small JSON command `{ "command": "evaluate" }`.
3. The Python worker computes all event or bin weights and writes them into the shared `weights` array.
4. The common worker layer updates the shared weight vector.
5. Each `ExternalWeightDispatcher` returns the weight corresponding to its event or bin index.

The large numerical arrays are therefore never serialized to JSON and are not written to disk. JSON is only used as a small control protocol for metadata and commands.

The current cache-manager backend does not support `ExternalWeight` dials. When such dials are present, GUNDAM falls back to the CPU propagation path.
