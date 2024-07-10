## DialSet

[< back to parent (ParameterSet)](./ParameterSet.md)

[< back to parent (ParameterSet)](./Parameter.md)

### Description

### Config options

| Option                 | Type         | Description                                                     | Default |
|------------------------|--------------|-----------------------------------------------------------------|---------|
| applyOnDataSets        | list(string) | list (regex) of datasets the corresponding dials will apply to  | `["*"]` |
| printDialsSummary      | bool         | extra verbose                                                   | false   |
| parametersBinningPath  | string       | create a dedicated norm dial according to the parameter binning |         |
| dialsDefinitions       | json         | dials config                                                    |         |
| dialsType              | string       | {Norm, Normalization, Spline, Graph}                            |         |
| dialSubType            | string       | {not-a-knot, natural, catmull-rom, light, monotonic} [1]        | empty   |
| applyCondition         | string       | formula condition that applies on every dial of the set         |         |
| applyConditions        | json         | config gathering multiple formulas                              |         |
| minDialResponse        | double       | cap dial response                                               |         |
| maxDialResponse        | double       | cap dial response                                               |         |
| useMirrorDial          | bool         | enable dial mirroring along the edges                           | false   |
| mirrorLowEdge          | double       | low edge where mirroring applies                                |         |
| mirrorHighEdge         | double       | upper edge where mirroring applies                              |         |
| allowDialExtrapolation | bool         | evaluate dials even out of boundaries                           | false   |

[1] The values for the dialSubType depend on the value of dialsType.  Specifically:

* Norm: This is a normalization parameter, and the subtype is currently ignored
* Graph: This does linear interpolation between points.  The subtype
      values are ROOT, or light.
  - ROOT: A ROOT TGraph object is used
  - light: A gundam LightGraph object is used.
* Spline: A spline controlled by input knots provided as a
      graph.  In several cases a GeneralSpline, or UniformSpline
      object may be used.  The choice depends on the spacing of the
      input points.  UniformSpline assumes regular point spacing, and
      GeneralSpline does not require that.  The subtype meanings are:
  - ROOT: A ROOT TSpline3 without constraints is used.  This results
      the slopes being estimated using the Not-A-Knot criteria.
  - not-a-knot: Use a GeneralSpline, or UniformSpline with the slopes
      calculated using the not-a-knot criteria.
  - natural: Use a GeneralSpline with the slopes calculated using the
      natural spline criteria
  - catmull-rom: Estimate the slopes using the Catmull-Rom criteria.
      This uses the average slopes for the points before and after.
  - monotonic: Apply the monotonic critera to the slopes before they
      are used.  This applies to "not-a-knot", "natural" and
      "catmull-rom" splines.  The monotonic criteria cannot be applied
      the "ROOT" TSpline3.

* Surface: A surfaced controlled by a mesh of input knots.  The mesh
      is usually provided by a TH2 object (for a uniform grid), but
      the mesh could be specified by a TGraph2D object The subtyupe
      meanings are:
  - Bilinear : Bilinear interpolation with a regular grid of knots.
  - Bicubic : Bicubic interpolation (i.e. a 2D spline) with a regular
      grid of knots.

### applyConditions options

| Option                            | Type               | Description                                                      | Default |
|-----------------------------------|--------------------|------------------------------------------------------------------|---------|
| exp / expression / var / variable | string             | formula condition that applies on every dial of the set          |         |
| allowedRanges                     | list(pair(double)) | list of ranges (min, max) where the dials of the set apply       |         |
| excludedRanges                    | list(pair(double)) | list of ranges (min, max) where the dials of the set don't apply |         |
| allowedValues                     | list(double)       | list of values where the dials of the set apply                  |         |
| excludedValues                    | list(double)       | list of values where the dials of the set apply                  |         |




### dialsDefinitions options


| Option              | Type   | Description                                                        | Default |
|---------------------|--------|--------------------------------------------------------------------|---------|
| isEnabled           | bool   | use the definition of these dials                                  | true    |
| dialLeafName        | string | fetch dial from the dataset TTree with the corresponding leaf name |         |
| binningFilePath     | string | binning definition of a set of spline (along with `dialsFilePath`) |         |
| dialsFilePath       | string | root file containing the set of dials                              |         |
| dialsList           | string | path within root file to the list of dials                         |         |
| dialsTreePath (old) | string | tree name where the dials are stored                               |         |
| dialSubType         | string | the specific form of the dial (values depend on the dial type)     |         |
