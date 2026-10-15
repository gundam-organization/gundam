## Sample

[< back to parent (SampleSet)](./SampleSet.md)

### Description

### Config options


| Option        | Type         | Description                               | Default |
|---------------|--------------|-------------------------------------------|---------|
| **name**      | string       | sample name                               |         |
| isEnabled     | bool         | include the sample in the propagator      | true    |
| binning       | string       | path to binning definition file           |         |
| selectionCuts | string       | tree selection formula                    |         |
| datasets      | list(string) | list of dataset will populate this sample | `{"*"}` |
| mcNorm        | double       | MC sample global normalisation            | 1       |
| dataNorm      | double       | data sample global normalisation          | 1       |


## Binning response

The optional `binningResponse` distributes each event weight continuously over
nearby bins. It is disabled by default; omitting it or setting `isEnabled: false`
keeps the existing single-bin filling.

```yaml
name: reconstructedEnergy
useDynamicBinning: true
binning:
  binningDefinition:
    - {name: energy, edges: [0, 1, 2.5, 3]}
binningResponse:
  isEnabled: true
  variables:
    - {name: energy, width: 0.2}
```

`width` is the half-width of the transition, in the variable's units. At the
internal boundary `energy = 1`, the transfer takes place over `[0.8, 1.2]`.
Widths are fixed per variable, including when bins have unequal sizes.

For each axis the response integrates the compact kernel
`K(u) = 15/16 * (1 - u*u)^2` for `abs(u) < 1`, and zero elsewhere.
Its cumulative function is the quintic smoothstep, so bin fractions are twice
continuously differentiable. Multidimensional responses multiply the integrals
along the configured axes. Fractions are nonnegative and normalized per event;
the sum of its contributions equals its current weight to floating-point
precision. Far from internal boundaries an event contributes to exactly one bin.

The bins at each outer boundary extend their response to infinity. Events outside
the nominal range of a smoothed variable therefore contribute to the edge bins;
they are retained during loading. This preserves weight and continuity at the
outer boundaries as well. This behavior differs intentionally from hard binning,
which excludes out-of-range events.

Unlisted variables retain their exact bin selection. Condition variables cannot
be smoothed. All retained events must match a bin slice in the unsmoothed
variables; otherwise response evaluation throws an error. Use sample selections
to exclude unwanted categories. Within each slice defined by identical
unsmoothed edges, the bins must tile a finite rectangular domain without holes
or overlaps. Every bin must use the same set of variables.

Response fractions are cached with the histogram's event lists. After changing
stored variables, explicitly call `sample.updateEventBinning()` (with
`useDynamicBinning: true`) or `propagator.updateEventSampleBinning()`, then
propagate to refill the histograms. Propagation alone does not reassign bins.
Binning responses also work for fixed samples and are then computed at loading.
The parameters and dials are independent of this configuration.

The Python histogram API exposes `isBinningResponseEnabled()` and
`evaluateBinningResponse(event)`, returning entries with `binIndex` and
`fraction`. `event.getIndices().bin` remains the geometric bin index and can be
`-1` even when the response contributes to an edge bin.

Bin MC errors accumulate `(eventWeight * fraction)^2`. A shared event induces
correlations between bins; these covariances are not stored in `BinContent`.
Event MC throws draw once per event. Independent bin statistical throws are
unsupported with binning responses and raise an error, since their outcomes
cannot in general be represented by the shared event weights. The cache manager
has not been adapted to this feature; use the CPU histogram filling path.

`tests/regular-tests/218BinningResponse.py` produces PNG/PDF comparisons of hard
and smooth responses, sums of fractions and numerical derivatives, a quadratic
objective plot, and a CSV of the scan. It covers 1D and 2D binning, unequal bin
sizes, categorical slices, external boundaries and several thread counts. Its
Python environment needs `uproot`, `numpy` and `matplotlib`.
