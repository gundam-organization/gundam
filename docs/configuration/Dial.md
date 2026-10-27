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
| dialsTreePath          | string       | Deprecated: tree name where the dials are stored                       |         |

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
    dialInputList:
      - {name: par0}
      - {name: par1}
    options:
      nodes:
        - name: first
          dialType: Formula
          dialInputList: [{name: value, inputIndex: 0}]
          options:
            formulaStr: "[value] * [value]"
        - name: second
          dialType: Formula
          dialInputList: [{name: value, inputIndex: 1}]
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

`inputIndex` is a zero-based index into the composite's own `dialInputList`.
`dialResponse` names another node in the same graph. Formula nodes also accept
`eventVariable`, naming a loaded event variable. Exactly one of these three
keys is required for each node input. Inputs may mix these sources.
Formula inputs require a unique `name`, used as `[name]` in the expression.
Other dial types retain positional argument order. Node declaration order
does not matter; shared dependencies are evaluated once per graph evaluation.
Node names must be non-empty and unique, dependencies must exist, and every
node must contribute to `output`. Cycles are rejected.

At parameter-set level, give the composite an explicit `dialInputList`. When
attached to a single parameter, omitting this list exposes that parameter as
input 0. Each node needs an explicit, non-empty `dialInputList`.

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
| `Composite` | Another graph; its input indices refer to this node's input list |
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

For example, `{name: offset, parameterName: delta}` binds `[offset]` to `delta`.
Input declaration order does not determine ROOT parameter order: names are
resolved once when configuring the dial. Positional `x[i]` expressions and
parameter references using only `{name: delta}` are unsupported for Formula.

Inside a composite, use `{name: delta, inputIndex: 0}` and
`{name: alpha, inputIndex: 1}` for its parameter inputs. Event references remain
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
