## Dial Definition

[< back to parent (ParameterSet)](./ParameterSet.md)

[< back to parent (ParameterSet)](./Parameter.md)

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
| dialSubType            | string       | See below                                                              | empty   |
| isEnabled              | bool         | Dial is not used if this is false                                      | true    |
| applyOnDataSets        | list(string) | list (regex) of datasets the corresponding dials will apply to         | `["*"]` |
| applyCondition         | string       | Only apply this dial with formula evaluates to non-zero                | empty   |
| applyConditions        | list(json)   | Multiple condition formulas (only apply dial if all are true) formulas | n/a     |
| minDialResponse        | double       | Deprecated, still supported with a warning. Use `dialResponseLimits`.   | -inf    |
| maxDialResponse        | double       | Deprecated, still supported with a warning. Use `dialResponseLimits`.   | inf     |
| dialResponseLimits     | range        | Response bounds as `[min, max]`, `{min: ...}`, or `{max: ...}`           | existing response defaults |
| enableResponseSmoothing | bool       | Apply centered C² transitions at finite response bounds in the original engine | false |
| dialResponseSmoothingWidth | positive number or range | Transition half-width, shared or separate for each bound | unset |
| binningFilePath        | string       | binning definition of a set of spline (along with `dialsFilePath`)     |         |
| useMirrorDial          | bool         | enable dial mirroring along the edges                                  | false   |
| mirrorLowEdge          | double       | low edge where mirroring applies                                       |         |
| mirrorHighEdge         | double       | upper edge where mirroring applies                                     |         |
| allowDialExtrapolation | bool         | Allow splines to be extrapolated past the first and last knots         | false   |
| printDialsSummary      | bool         | extra verbose                                                          | false   |
| definitionRange        | pair(double) |                                                                        |         |
| mirrorDefinitionRange  | pair(double) |                                                                        |         |
| dialInputList          | list(json)   | List of parameters for this dial                                       | empty   |
| dialsTreePath          | string       | Deprecated: tree name where the dials are stored                       |         |

### Smooth response limits

The original propagation engine can replace hard response clamps with centered
quartic transitions. This operates on the evaluated dial response, independently
of the direction in which the parameter changes, and does not modify input knots.
Accelerated backends and the legacy CacheManager do not implement this option.

```yaml
dialType: Spline
dialBranchData: graph
dialResponseLimits: [0.0, 2.0]
enableResponseSmoothing: true
dialResponseSmoothingWidth: 0.1
```

A scalar width applies to both bounds. Separate widths can be specified as
`[0.2, 0.05]` or `{min: 0.2, max: 0.05}`. The first entry always belongs to the
lower bound; widths are never reordered by magnitude. For a single bound, for
example `dialResponseLimits: {max: 2.0}`, `{max: 0.05}` is sufficient.

All supplied widths must be finite and strictly positive, including widths for
absent bounds. Set `enableResponseSmoothing: false` to disable smoothing rather
than using a zero width. Enabling smoothing requires a width for each finite
bound. Widths for absent or infinite bounds are otherwise unused. With two finite
bounds, the sum of their widths must not exceed `max - min`; touching transition
regions are allowed, overlapping regions are rejected.

For a lower limit `L` and width `c`, the response is exactly `L` for raw weights
`w <= L - c`, unchanged for `w >= L + c`, and otherwise:

```text
t = (w - L + c) / (2c)
response = L + c * t³ * (2 - t)
```

The upper transition is the reflection of the lower transition. The value and
first two derivatives match at the transition endpoints, and the slope stays
between zero and one. In particular, a raw weight equal to a lower limit of zero
returns `3c/16`, reaching exactly zero at `w = -c`. This smooths clamp junctions;
it does not guarantee removal of kinks already present in the dial itself.

`dialResponseLimits` also works without smoothing. When supplied, it replaces the
default response bounds: an omitted bound is unbounded. The existing
`minDialResponse` and `maxDialResponse` options are deprecated but remain
supported with a warning, including with smoothing enabled. Old configurations
retain their response bounds and hard-clamping behavior unless smoothing is
explicitly enabled. The older `minimumSplineResponse` alias also remains readable.
If combined with `dialResponseLimits`, explicitly supplied
legacy values must match the corresponding limits; conflicts are rejected.
Omitting `enableResponseSmoothing` preserves hard clamping even if a width is
provided.

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
the [tableConfig](TabulatedDials.md) option.

#### `dialType: Kriged`

| Option                 | Type         | Description                                                            | Default |
|------------------------|--------------|------------------------------------------------------------------------|---------|
| tableConfig            | json         | Definition of functions to call for Kriged dials                      | empty   |

Use a precalculated table of weights.  The table may be refilled for each
iteration of the fitter. The method of filling the table is defined using
the [tableConfig](KrigedDials.md) option.

#### `dialType: ExternalWeight`

| Option            | Type      | Description                                             | Default |
|-------------------|-----------|---------------------------------------------------------|---------|
| type              | string    | Worker implementation; currently `PythonWorker`         |         |
| inputEventVarList | list(str) | Optional event variables exposed to the worker          | empty   |
| workerConfig      | json      | Configuration specific to the selected worker           |         |

Use an external worker to calculate one weight per selected event.
`PythonWorker` is the current implementation; other worker types can use the
same shared-memory inputs, parameters, and output weights. JSON is only used
as a lightweight control protocol. See [ExternalWeightDials](ExternalWeightDials.md).

#### `dialType: Formula`

Apply a weight using a ROOT formula.  This is not fully supported since it
cannot be efficiently applied with GPU acceleration.

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
