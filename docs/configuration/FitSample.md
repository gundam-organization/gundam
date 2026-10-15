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


## Binning variables

All bins within a binning must define exactly the same set of variable names.
Their order may differ, and bin widths and bounds may vary. A variable cannot
appear more than once in a bin. Different samples may use different sets of
variables, provided each sample's binning is internally consistent.

This validation applies to every `BinSet`, independently of dynamic binning or
smoothing, and covers JSON/YAML definitions and text files with multiple
`variables:` sections. Inconsistent definitions are rejected during
configuration, before checking bin overlaps, with the offending bin index and
its expected and actual variable names.

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
Absolute widths are fixed per variable, including when bins have unequal sizes.

For an absolute width, the response integrates the compact kernel on each axis
`K(u) = 15/16 * (1 - u*u)^2` for `abs(u) < 1`, and zero elsewhere.
Its cumulative function is the quintic smoothstep, so bin fractions are twice
continuously differentiable. Multidimensional responses multiply the integrals
along the configured axes. Each bin uses its actual bounds, including at the
outer contour. Fractions are nonnegative and are not renormalized over the
physical bins. The remaining fraction is tracked separately:

```text
outsideFraction = 1 - sum(binFractions)
```

For every event, physical-bin contributions plus its outside contribution equal
its current weight to floating-point precision. Far inside a sufficiently large
bin, the event contributes only to that bin. Far outside the domain, its entire
weight goes to `outside`. Events outside the nominal domain are retained during
loading when the response is enabled.

The outside content is a diagnostic quantity, separate from the physical bins.
It does not add a bin to the histogram bin list, plots of physical bins, or the
likelihood. The predicted rate inside the domain can therefore change smoothly
as events cross its external contour. `Sample::getSumWeights()` still sums the
full event weights, including outside contributions; `getNbBinnedEvents()` counts
geometric membership, using `event.getIndices().bin`.

Each individual bin must be rectangular and use the same set of variables.
The union of bins need not be rectangular: rows of different lengths and
L-shaped domains are supported. Uncovered regions, including gaps, contribute
to `outside`. Overlapping bins remain unsupported.

Unlisted variables retain their exact bin selection. Condition variables cannot
be smoothed. An event matching no interval or category in the unsmoothed
variables has `outsideFraction = 1`. Changes to unsmoothed variables and sample
selections can still introduce discontinuities; continuity of the response only
applies to the smoothed axes.

### Relative widths

Use `widthFraction` instead of `width` to scale transitions with the binning:

```yaml
binningResponse:
  isEnabled: true
  variables:
    - {name: energy, widthFraction: 0.1}
```

Each variable must specify exactly one of `width` and `widthFraction`. A relative
fraction must be finite and in `(0, 0.5]`. Different axes can mix absolute and
relative widths.

In a contiguous 1D binning, a shared boundary has half-width
`widthFraction * min(leftBinWidth, rightBinWidth)`. An external boundary uses its
adjacent bin width. Both sides use the same transition, built from the quintic
cumulative kernel. The two bounds of a bin may therefore have different widths:

```text
binFraction = F((upperEdge - x) / upperWidth)
            - F((lowerEdge - x) / lowerWidth)
```

Transition bands cannot overlap within an elementary interval because the
fraction is at most `0.5`. In 1D an event contributes to at most two physical
bins, to floating-point precision. This restriction does not apply to absolute
widths, which can distribute an event over more bins. Both modes retain the
`outside = 1 - sum(binFractions)` convention and a C2 response on smoothed axes.

For multidimensional or disconnected domains, the relative scale is defined
using the sorted, distinct bin edges projected onto each axis. A boundary uses
the smaller neighboring edge spacing, or the only spacing at an extremity.
These common transitions preserve a nonnegative partition even when neighboring
rows have different lengths or staggered edges. Intermediate projected edges
can make a transition narrower than a fraction of the original large bin.
Gaps also participate in the spacing calculation and contribute to `outside`.

Disjoint categories and other nonoverlapping selections in unsmoothed variables
have independent projected edges. Overlapping selections, including transitive
overlaps, share their projected edges to keep their transitions compatible.
When all smoothed axes use relative widths, at most `2^D` physical bins can
receive a nonzero contribution in D dimensions, to floating-point precision.

All transition widths are computed at initialization. No local-width
interpolation or bin-size lookup occurs in the event loop. The absolute-width
path keeps its original arithmetic and interval layout; the batch selects its
evaluation routine once per worker, outside the event loop.

Response fractions are cached with the histogram's event lists. After changing
stored variables, explicitly call `sample.updateEventBinning()` (with
`useDynamicBinning: true`) or `propagator.updateEventSampleBinning()`, then
propagate to refill the histograms. Propagation alone does not reassign bins.
Binning responses also work for fixed samples and are then computed at loading.
The parameters and dials are independent of this configuration.

The Python histogram API exposes:

- `isBinningResponseEnabled()`;
- `evaluateBinningResponse(event)`, returning only physical-bin entries with
  `binIndex` and `fraction`;
- `evaluateEventBinningResponse(event)`, returning `binFractions` and
  `outsideFraction`;
- `getOutsideBinContent()`, with `sumWeights` and `sqrtSumSqWeights`, updated
  alongside the physical-bin contents during histogram filling.

`event.getIndices().bin` remains the geometric bin index. It can be `-1` while
some response still contributes to nearby physical bins. With the response
disabled, histogram filling remains unchanged and outside content is not
accumulated.

Bin MC errors accumulate `(eventWeight * fraction)^2`. A shared event induces
correlations between bins; these covariances are not stored in `BinContent`.
Outside MC errors use the same convention. Event MC throws draw once per event,
including events contributing only outside. Independent bin statistical throws are
unsupported with binning responses and raise an error, since their outcomes
cannot in general be represented by the shared event weights. The cache manager
has not been adapted to this feature; use the CPU histogram filling path.

`tests/regular-tests/218BinningResponse.py` produces PNG/PDF comparisons of hard
and smooth responses, sums of fractions and numerical derivatives, a quadratic
objective plot, and a CSV of the scan. It covers 1D and 2D binning, unequal bin
sizes, categorical slices, external boundaries, L-shaped domains and several
thread counts. The plots include the outside response and conservation of
physical-bin plus outside weights. A separate 2D map shows the response across
the full L-shaped domain. Its Python environment needs `uproot`, `numpy` and `matplotlib`.

`tests/regular-tests/220BinningWidthFraction.py` checks the width options, the
1D two-bin limit, numerical derivatives, mixed modes, independent categories,
and staggered L-shaped domains across thread counts. It also plots a comparison
of absolute and relative widths around a narrow bin.

### CPU parallelism

Binning responses are evaluated once per event, partitioning the event list over
the workers. Each worker reuses its own evaluation buffer and contribution list.
Variable indices, interval definitions and inverse widths are prepared outside
the event loop; each variable value is read once per event.

Workers first count their contributions per bin. Prefix offsets then give each
worker a disjoint part of each bin list, which it fills without locks or atomic
updates. Event order within a bin is preserved across thread counts. Temporary
storage scales with the number of nonzero contributions plus workers times bins,
and its allocations are reused between rebinning calls.

The histogram refill distributes physical bins and the separate outside content
over workers and uses the
cached fractions. Changing only event weights therefore does not recompute the
response. Rebinning still scans the bin definitions per event; a spatial lookup
could further reduce this cost for large binnings. Refill parallelism is limited
by the number of bins and can be uneven when their populations differ.

For repeated updates, prefer `propagator.updateEventSampleBinning()` in Python:
it reuses the propagator's thread pool. The standalone
`sample.updateEventBinning()` creates a temporary pool for each call.

The optional benchmark helper measures rebinning separately from propagation
(reweighting, histogram filling and likelihood evaluation):

```bash
python /path/to/tests/regular-tests/218BinningResponse-benchmark.py \
  --events 100000 --bins 4 64 --threads 1 2 4 8 --repeat 7
```

Pass `--width-fraction 0.25` to benchmark relative widths. Without this option,
the benchmark uses the absolute-width path.

Run it in the test Python environment with the intended compiled GUNDAM module on
`PYTHONPATH`. The JSON output includes the loaded module path, median timings,
physical-bin and outside contents and MC errors. Compare builds with the same
inputs, compiler options and an otherwise idle machine; there is no timing threshold in the test suite.
