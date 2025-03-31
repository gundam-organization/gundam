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

### Config options that apply to specific types of dials

| Option                 | Type         | Description                                                            | Default |
|------------------------|--------------|------------------------------------------------------------------------|---------|
| dialLeafName           | string       | fetch dial from the dataset TTree with the corresponding leaf name     |         |
| dialsFilePath          | string       | root file containing the set of dials                                  |         |
| dialsList              | string       | path within root file to the list of dials                             |         |
| tableConfig            | json         | Definition of functions to call for Tabular dials                      | empty   |

### Configuration for different dial types

#### `dialType: Normalization`

| Option                 | Type         | Description                                                            | Default |
|------------------------|--------------|------------------------------------------------------------------------|---------|
| parametersBinningPath  | string       | create a dedicated norm dial according to the parameter binning        |         |

Treat the parameter as a simple weight.  In this case, the parameter must be bounded to always be positive.

#### `dialType: Spline`

| Option                 | Type         | Description                                                            | Default |
|------------------------|--------------|------------------------------------------------------------------------|---------|
| dialLeafName           | string       | fetch dial from the dataset TTree with the corresponding leaf name     |         |
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
| dialLeafName           | string       | fetch dial from the dataset TTree with the corresponding leaf name     |         |
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
| dialLeafName           | string       | fetch dial from the dataset TTree with the corresponding leaf name     |         |
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
| tableConfig            | json         | Definition of functions to call for Tabular dials                      | empty   |

Use a precalculated table of weights.  The table may be refilled for each
iteration of the fitter. The method of filling the table is defined using
the [tableConfig](TabulatedDials.md) option.

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
