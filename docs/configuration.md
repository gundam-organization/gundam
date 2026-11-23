# Configuration File Reference

## Configuring GUNDAM

GUNDAM is configured using YAML (Json) input files that describe and control every aspect of the job.  For the most part, the input files are directly handled as standard YAML, with one minor exception to include files into the configuration.

### Best practices: Including other files into the configuration.

The recommended practice is to break the configuration into several input
files that reflect the logical structure and to include them using the
include file mechanism.

The GUNDAM libraries start by reading the entire configuration in to
memory, and then it is interpreted into a json structure according to the
[usual rules](https://www.yaml.org) with one exception.  If the value of a
string is found to end in the extension ".yaml", then the string is treated
as a filename relative to the current working directory to be included at
that point, and the contents of the file is substituted into the input
stream.  For instance

```yaml
fitterEngineConfig: "./inputs/engineConfiguration.yaml"
```

will read the contents of the "./inputs/engineConfiguration.yaml" file as a `Json` object add it to the full `Json` object as a child of the "fitterEngineConfig" field.  This provides convenient ways to factor the full configuration into components.

```yaml
minGundamVersion: 2.0.0
fitterEngineConfig:
    minimizerConfig:
        type: RootMinimizer
        minimizer: Minuit2
    likelihoodInterfaceConfig:
        jointPropabilityConfig:
            type: BarlowLLH
```
could be broken into three files:

- topConfig.yaml
```yaml
minGundamVersion: 2.0.0
fitterEngineConfig:
    minimizerConfig: "./minimizerConfig.yaml"
    likelihoodInterfaceConfig: "./likelihoodConfig.yaml"
```
  - minimizerConfig.yaml
```yaml
type: RootMinimizer
minimizer: Minuit2
```

  - likelihoodConfig.yaml
```yaml
jointPropabilityConfig:
    type: BarlowLLH
```

### Top Level Configuration Options.

| Option | Type | Description | Default |
|------------------------------------------|--------|------------------------------------------------|---------|
| [fitterEngineConfig](#fitterengineconfig)    | json   | FitterEngine config                            |         |
| minGundamVersion (the format is "v.r.p") | string | gundamFitter will stop if the version is lower |         |
| outputFolder                             | string | Folder where the output file is written        | ./      |

## FitterEngineConfig

[< back to parent (gundamFitter)](run-gundam.md#gundamfitter)

### Description

### Config options

| FitterEngine Options                                | Type         | Description                                                                   | Default  |
|-----------------------------------------------------|--------------|-------------------------------------------------------------------------------|----------|
| [likelihoodInterfaceConfig](#likelihoodinterfaceconfig) | json         | Configure the likelihood calculation.                                         | required |
| [minimizerConfig](minimizers.md#configminimizerinterface)            | json         | MinimizerInterface config                                                     | required |
| [parameterScannerConfig](#parameter-scanner-config)       | json         | Scan parameters config                                                        |          |
| scanConfig                                          | json         | Deprecated. See parameterScannerConfig.                                       |          |
| propagatorConfig                                    | json         | Deprecated. See likelihoodInterfaceConfig/propagatorConfig                    |          |
| mcmcConfig                                          | json         | Deprecated. See minimizerConfig                                               |          |
| engineType                                          | string       | Deprecated. See minimizerConfig/type                                          |          |
| enablePreFitScan                                    | bool         | Run fit parameter scan right before the minimization                          | false    |
| enablePostFitScan                                   | bool         | Run fit parameter scan right after the minimization                           | false    |
| enablePreFitToPostFitLineScan                       | bool         | Run scan between the prefit and postfit parameters                            | false    |
| generateSamplePlots                                 | bool         | Draw sample histograms according to the PlotGenerator config                  | true     |
| allParamVariations                                  | list(double) | List of points to perform individual parameter variation                      |          |
| enablePca / runPcaCheck / fixGhostFitParameters     | bool         | Dangerous: Fix parameter if the effect on stat LHH is lower than `pcaDeltaChi2Threshold` | false    |
| pcaThreshold                                        | double       | LLH threshold for PCA                                                         | 1E-6     |
| fixGhostEigenParametersAfterFirstRejected           | bool         | Fix all next parameters once PCA has been triggered (dev)                     | false    |
| throwMcBeforeFit                                    | bool         | Push MC parameter away from their prior before fitting                        | false    |
| throwMcBeforeFitGain                                | int          | Scale throws for MC parameters                                                | 1        |
| customFitParThrow*                                  | list         | Use the custom thrown values for parameters (dev)                             |          |
| scaleParStepWithChi2Response                        | bool         | Use LLH profile to scale parameter step size (dev)                            | false    |
| parStepGain                                         | bool         | Boost step value with `scaleParStepWithChi2Response` (dev)                    | 0.1      |
| savePostfitEventTrees                               | bool         | Save event tree after the fit.                                                | false    |

### JSON sub-structures

#### customFitParThrow options

| Options         | Type | Description                           | Default |
|-----------------|------|---------------------------------------|---------|
| **parIndex**    | int  | Parameter index to move               |         |
| **nbSigmaAway** | int  | Custom throw value (in unit of sigma) |         |

## LikelihoodInterfaceConfig
[< back to parent (fitterEngineConfig)](#fitterengineconfig)

### Description

Define how the likelihood will be calculated.  This includes defining the
datasets that will be used for the model and measurement.  It also includes
defining the "propagator" that will apply the effect of fitting parameters
to the model.  The comparison between the measurement and model is defined
using `jointProbabilityConfig`.

### Config options

| Likelihood Interface Options        | Type | Descriptions                                | default   |
|-------------------------------------|------|---------------------------------------------|-----------|
| [propagatorConfig](#propagator-configuration) | Json | Configure the parameter propagation         |           |
| [datasetManagerConfig](#datasetmanager) | Json | Configure the input datasets | |
| datasetList | Json | Deprecated: use `datasetManagerConfig.datasetList` | |
| jointProbabilityConfig              | Json | Configure the joint probability calculation | BarlowLLH |
| plotGeneratorConfig                 | Json |                                             |           |
| enableStatThrowInToys               | bool |                                             |           |
| gaussStatThrowInToys                | bool |                                             |           |
| enableEventMcThrow                  | bool |                                             |           |

### `jointProbabilityConfig`

The `jointProbabilityConfig` block is forwarded to the selected joint
probability implementation.

#### Common options

| Joint Probability Options             | Type | Description                                                                                                                              | default |
|---------------------------------------|------|------------------------------------------------------------------------------------------------------------------------------------------|---------|
| ignoreBinsWithZeroPredictionAtPrior   | bool | If `true`, bins with a null model prediction at the prior are ignored in the likelihood evaluation. A warning is printed for each bin disabled this way. | `false` |

This option is intended for analyses where the user-defined binning contains
bins that are meant to stay empty, for example with square binning layouts
that include unreachable phase-space regions.

### `datasetManagerConfig`

```yaml
likelihoodInterfaceConfig:
  datasetManagerConfig:
    datasetList:
      - name: input
        model:
          tree: events
          filePathList: [input.root]
```

The former `datasetList` directly under `likelihoodInterfaceConfig` is still
read with a relocation warning. If both lists are present, the list in
`datasetManagerConfig` takes precedence, including an explicitly empty list.
The older `propagatorConfig.dataSetList` location remains supported as a fallback.

## Monte Carlo statistical likelihoods

### BarlowBeestonLiteGaussian

```yaml
jointProbabilityConfig:
  type: BarlowBeestonLiteGaussian
```

The current model uncertainty `sqrt(sum(w^2))` is used at every evaluation.
There is no option to freeze nominal uncertainties, no nominal cache, and no
BANFF compatibility options. Scalar and sample calls use the same convention:
the uncertainty argument is a standard deviation, not a variance.
The historical `BarlowLLH_BANFF_OA2021` implementation remains available unchanged.
Legacy thresholds, fractional-error caps and finite replacements of infinity
are not reproduced.

For observation n, prediction mu and MC standard deviation sigma, the objective is

    min over x >= 0: 2*(x-n+n*log(n/x)) + (x-mu)^2/sigma^2.

For mu>0, x=beta*mu gives Conway's equations (10)-(11). The absolute-mean
parameterization also defines the continuous extension at mu=0 with sigma>0.
For n=0 the logarithmic term is zero; for sigma=0 the MC mean is fixed to mu.
An empty data/MC bin contributes zero. Invalid inputs (negative or nonfinite)
return positive infinity. Impossible observations with zero fixed expectation
also return infinity; undefined derivatives return NaN. There is no logging or
exception policy option. These are nonnegative-weight models.

`evaluate` returns the objective and analytic partial derivatives with respect to
mu and sigma using the envelope theorem. Sample gradients respect ignored bins. A full parameter gradient combines these partials with derivatives
of the propagated histogram. No finite-difference mutation or shared scratch
buffer is used.

Reference: J. S. Conway, *Incorporating Nuisance Parameters in Likelihoods for
Multisource Spectra*, [arXiv:1103.0354](https://arxiv.org/abs/1103.0354), section 5.

### Exact Barlow-Beeston solver

`BarlowBeeston::evaluateSources(n, sources, fixedPrediction)` profiles the model

    n ~ Poisson(fixedPrediction + sum_j scale_j * A_j)
    count_j ~ Poisson(A_j).

Each `Source` contains an observed unweighted integer MC `count` and its current
nonnegative `scale`. The optional fixed prediction has no MC uncertainty.
The result contains the saturated deviance, fitted prediction, fitted source
means and envelope derivatives with respect to source scales.

A monotonic one-dimensional equation is solved by bisection. Empty MC sources,
including an active solution on the boundary associated with the largest scale,
are retained. In a degenerate boundary solution the allocation between identical
empty sources is nonunique; reported scale derivatives follow the selected
allocation and are not a unique ordinary gradient at that point.

**Fit integration needs per-source inputs.** The existing histogram interface
only supplies sum(w) and sqrt(sum(w^2)); these do not determine the individual
counts and scales. Consequently `BarlowBeeston` refuses scalar/histogram evaluations
with nonzero uncertainty rather than silently constructing an effective single
source. With zero uncertainty it evaluates the fixed-mean Poisson deviance.
The new source API implements the complete statistical model, but is not yet
connected to source decomposition during propagation. No backend changes are made.

Reference: R. Barlow and C. Beeston, *Fitting using finite Monte Carlo samples*,
Comput. Phys. Commun. 77 (1993) 219-228,
[doi:10.1016/0010-4655(93)90005-W](https://doi.org/10.1016/0010-4655(93)90005-W).

### BarlowBeestonLitePoisson

```yaml
jointProbabilityConfig:
  type: BarlowBeestonLitePoisson
```

Implements the improved Dembinski-Abdelmotteleb approximation. With unweighted
Poisson data and the current weighted-MC moments, it uses

    a_eff = mu^2 / variance
    beta = (n + a_eff)/(mu + a_eff)
    q = PoissonDeviance(n, beta*mu) + PoissonDeviance(a_eff, beta*a_eff).

It keeps one nuisance per bin and an analytic profile, replacing the Gaussian
penalty by a Poisson deviance. It preserves asymmetry at low effective MC counts
and agrees with the Gaussian penalty near beta=1. It remains an approximation
for a mixture of templates. It is exact for one unweighted Poisson template;
the effective-count extension to varying event weights is still approximate.
Improvements in fit bias or coverage are problem-dependent and require toy
validation; it does not solve bins with no simulated support automatically.

Reference: H. Dembinski and A. Abdelmotteleb, *A fast and stable approximate
maximum-likelihood method for template fits*,
[arXiv:2206.12346](https://arxiv.org/abs/2206.12346), especially equations (17)-(18).

The MC uncertainty is always current, as in `BarlowBeestonLiteGaussian`. Both
classes have no class-specific configuration options. `ignoreBinsWithZeroPredictionAtPrior`
is inherited from `JointProbabilityBase`.

`evaluate` returns analytic partial derivatives with respect to mu and sigma.
These include the dependence of a_eff on both inputs, which must be included
when differentiating weighted predictions. Products involving an extreme
a_eff or beta are evaluated using logarithmic scales to avoid intermediate
overflows. A series preserves small displacements near an Asimov minimum.

For sigma=0, the result is the ordinary Poisson deviance. The empty input
(n,mu,sigma)=(0,0,0) contributes zero; mu=0 with n>0 or sigma>0 returns infinity.
The latter has no valid effective-count parameterization for nonnegative MC
weights. Negative/nonfinite inputs return infinity, with NaN derivatives.
At (0,0,0), the prediction derivative is the one-sided fixed-sigma value 2.

### Shared numerical helper

`src/Utils/include/PoissonDeviance.h` is a numerical utility, not another configurable likelihood.
Its `GundamUtils::halfPoissonDeviance(n, mu)` returns

    mu - n + n*log(n/mu),

half the saturated Poisson deviance (the likelihood classes supply the factor 2).
It handles n=0 and uses a series or log1p near mu=n so that subtracting almost
equal numbers does not destroy the quadratic signal needed by Hesse and
gradient checks. An overload accepts the residual (mu-n)/n separately when
profiling would otherwise round away a small displacement. It is shared by
the exact Barlow-Beeston solver and both lite likelihoods.

## Propagator Configuration

[< back to parent (likelihoodInterfaceConfig)](#likelihoodinterfaceconfig)

### Description

The propagator handles the engine in charge of applying parameters to the
samples.  It owns the parameters, the samples, and the data loader.

### Config options

| Option                                                      | Type       | Description                                                                                | Default |
|-------------------------------------------------------------|------------|--------------------------------------------------------------------------------------------|---------|
| [sampleSetConfig](#sampleset)                             | json       | Configure the samples that will be compared data vs model                                  |         |
| [parametersManagerConfig/parameterSetList](parameters-and-dials.md#parametersetlist) | json       | Configure the parameters that will affect the model                                        |         |
| [parameterInjection/parameterSetList](parameters-and-dials.md#parametersetlist)      | list(json) | Override values for the parameters                                                         |         |
| showNbEventParameterBreakdown                               | bool       | Print the number of event affected by each parameter                                       | false   |
| showNbEventPerSampleParameterBreakdown                      | bool       | Print the number of event per sample affected by each parameter                            | false   |
| globalEventReweightCap                                      | double     | Will cap the weight applied by the parameters: evWeight = baseWeight * min(parWeight, cap) | nan     |
| debugPrintLoadedEvents                                      | bool       | If true then print contents for a set of events.                                           | false   |
| debugPrintLoadedEventsNbPerSample                           | int        | Number of events to dump in the debugPrintLoadedEvents output                              | 5       |
| devSingleThreadReweight                                     | bool       | Use a single thread for reweighting events with the CPU                                    | false   |
| devSingleThreadHistFill                                     | bool       | Use a single thread for filling the histograms with the CPU                                | false   |

## DatasetManager

[< back to parent (likelihoodInterfaceConfig)](#likelihoodinterfaceconfig)

Owns, configures and initializes the dataset definitions used to load model
and measurement events.

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| name | string | Optional identifier for configuration overrides | |
| datasetTemplateList | list | Named partial data dispenser configurations | Empty |
| datasetList | list | Dataset definitions, with their existing `name`, `model` and optional `data` fields | Empty |

### Dataset templates

`datasetTemplateList` holds named, partial data dispenser configurations. Template
fields are validated when the effective dispenser is configured.

```yaml
datasetManagerConfig:
  datasetTemplateList:
    - name: common
      tree: events
      filePathList: [input.root]
    - name: selection
      selectionCutFormula: "energy > 0"
  datasetList:
    - name: selectedEvents
      model:
        useTemplates:
          - name: common
          - name: selection
        tree: selected
      data:
        - name: measurement
          useTemplates:
            - name: common
          filePathList: [data.root]
      selectedDataEntry: measurement
```

Templates are applied in `useTemplates` order, followed by the dispenser's own
configuration, using the usual GUNDAM configuration override rules. A template's
`name` identifies the template and is not copied into the dispenser. References
are enabled by default; `isEnabled: false` skips the reference. Unknown enabled
template references and duplicate template names are errors. Templates do not
recursively use other templates.

`useTemplates` belongs to `model` or an entry of `data`, rather than to the
surrounding dataset definition. With `fromModel: true`, the effective model
configuration provides the base; the data templates and then the data entry's
own options override that base.

## DatasetDefinition

[< back to parent (Propagator)](#propagator-configuration)

### Description

Defines how the data should be loaded to the samples.
Each config entry refers to a data set.

### Config options

As an entry list

| Option                               | Type   | Description                                                         | Default |
|--------------------------------------|--------|---------------------------------------------------------------------|---------|
| name                                 | string | Name of the dataset                                                 |         |
| mc (see bellow)                      | json   | Config for the MC entry                                             |         |
| data (see bellow)                    | json   | Config list for the Data entries                                    |         |
| selectedDataEntry                    | string | Name of the 'data' entry that should be used to load data events    | Asimov  |
| selectedToyEntry                     | string | Name of the 'data' entry that should be used to perform a toy fit   | Asimov  |
| isEnabled                            | bool   | Specify if it should be considered during the runtime               | true    |
| showSelectedEventCount               | bool   | Show the number of events passing the selection cut for each sample | true    |
| devSingleThreadEventSelection        | bool   | Force the event selection to be performed in single thread          | false   |
| devSingleThreadEventLoaderAndIndexer | bool   | Force the event loading to be performed in single thread            | false   |


#### mc

| Option                  | Type                         | Description                                                                 | Default |
|-------------------------|------------------------------|-----------------------------------------------------------------------------|---------|
| tree                    | string                       | Name of the TTree containing the data in each file                          |         |
| selectionCutFormula     | formula                      | Global selection cut (should return 0 if not selected)                      |         |
| nominalTreeWeightFormula | formula                     | Formula that returns the base weight of a given event                       |         |
| filePathList            | list(string \| json)          | list of ROOT files containing the TTree; entries can attach friend TTrees   |         |
| additionalLeavesStorage | list(string)                 | list of variables to be stored in memory                                    |         |
| variablesTransform      | list(json)                   | Deprecated. Use `variableDict` entries with `evalFromLib` instead.          |         |
| variableDict            | list(json)                   | dictionary translating a leaf, formula, or shared-library output to a variable name |         |
| fromHistContent         | json                         | use hist bin content directly. This will create dummy events                |         |

Formula fields can be written as a single expression, a list of expressions, or a list of named expression entries:

```yaml
myFormula: "[var1] * 4"
myFormula: ["[var1] * 4", "[var2] + 1"]
myFormula:
  - { name: "part1", expr: "[var1] * 4" }
  - { name: "part2", expr: "[var2] + 1" }
```

`variableDict` entries can define either an `expr` or an `evalFromLib` block:

```yaml
variableDict:
  - { name: "myVar", expr: "treeBranch + 10." }
  - name: "myVarFromLib"
    evalFromLib:
      title: "optional title"
      libraryFile: "foo.so"
      inputList:
        - "[myVar]"
        - "TMath::Abs(mode)"
      messageOnError: "optional build hint"
```

`filePathList` keeps its historical list-of-strings syntax:

```yaml
filePathList:
  - "${DATA_DIR}/events.root"
```

It can also use named entries to attach ROOT friend trees. The `name` keys identify list entries for configuration overrides; friend names are ROOT aliases and should be used to qualify their branches in formulas.

```yaml
filePathList:
  - name: "run1"
    path: "${DATA_DIR}/events-run1.root:events"
    friendList:
      - name: "weights"
        path: "${DATA_DIR}/weights-run1.root:weights"
      - name: "truth"
        path: "${DATA_DIR}/truth-run1.root:truth"
  - name: "run2"
    path: "${DATA_DIR}/events-run2.root:events"
    isEnabled: true
    friendList:
      - name: "weights"
        path: "${DATA_DIR}/weights-run2.root:weights"
        isEnabled: true
      - name: "truth"
        path: "${DATA_DIR}/truth-run2.root:truth"

variableDict:
  - { name: "weight", expr: "weights.nominalWeight" }
```

To attach several friend trees from the same ROOT file, use `treeList`:

```yaml
friendList:
  - name: "xsec_syst_friends"
    path: "/path/to/friend/trees/file.root"
    treeList: ["friendTree1", "friendTree2"]

variableDict:
  - { name: "weight", expr: "friendTree1.nominalWeight * friendTree2.correction" }
```

With `treeList`, `name` identifies the group for configuration overrides, and each tree name becomes its ROOT friend alias. `path` must contain only the file path, without a `:tree` suffix. The list must be nonempty and contain nonempty tree names. Grouped entries can be mixed with single-tree entries; all resulting aliases must be unique within each main-file entry. Without `treeList`, `name` remains the alias as before.

Both main-file and friend entries accept `isEnabled` (default: `true`). Disabled main files are excluded from the chain; disabled friends are not attached. For grouped friends, `isEnabled` applies to every tree in the group. Every enabled file entry must declare the same enabled friend aliases. Each enabled friend tree must have exactly the same number and order of entries as its corresponding main tree. A branch with a unique name may be referenced without its alias, but `friendAlias.branch` is the recommended form because it remains unambiguous.


#### data

All options from MC

| Option                   | Type   | Description                                                                                                                     | Default |
|--------------------------|--------|---------------------------------------------------------------------------------------------------------------------------------|---------|
| name                     | string | Name of the data/toy entry                                                                                                      |         |
| fromModel                | bool   | Use config from model as a base for a data entry                                                                                | false   |
| useReweightEngine        | bool   | Force to load the dials while loading the data. Used for custom toys that use the reweight engine for generating the histograms | false   |
| evalModelAt              | json   | Specify which sets of parameters to eval the model and copy the events from. Parameter injector format                          |         |
| overridePropagatorConfig | json   | Use a custom set of config option that will override the propagator config. Any parameter can be changed from here              |         |

## SampleSet

[< back to parent (Propagator)](#propagator-configuration)

### Description

### Config options

| Option                          | Type       | Description                                               | Default    |
|---------------------------------|------------|-----------------------------------------------------------|------------|
| [fitSampleList](#sample) | list(json) | list of Sample config                                  |            |
| llhStatFunction                 | string     | name of the stat. LLH function                            | PoissonLLH |

### llhStatFunction entries

 - PoissonLLH
 - BarlowLLH
 - BarlowLLH_BANFF_OA2020
 - BarlowLLH_BANFF_OA2021 -> adds `llhConfig` option containing:

| Option               | Type | Description          | Default |
|----------------------|------|----------------------|---------|
| usePoissonLikelihood | bool | replica of the BANFF | false   |
| BBNoUpdateWeights    | bool | replica of the BANFF | false   |


 - Plugin (see `llhSharedLib` option)

| Option       | Type   | Description                                                     | Default |
|--------------|--------|-----------------------------------------------------------------|---------|
| llhSharedLib | string | path to the shared library (.so) with custom LLH function (dev) |         |

## Sample

[< back to parent (SampleSet)](#sampleset)

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

`tests/python/regular/350ModelBinningResponse.py` produces PNG/PDF comparisons of hard
and smooth responses, sums of fractions and numerical derivatives, a quadratic
objective plot, and a CSV of the scan. It covers 1D and 2D binning, unequal bin
sizes, categorical slices, external boundaries, L-shaped domains and several
thread counts. The plots include the outside response and conservation of
physical-bin plus outside weights. A separate 2D map shows the response across
the full L-shaped domain. Its Python environment needs `uproot`, `numpy` and `matplotlib`.

`tests/python/regular/360ModelBinningWidthFraction.py` checks the width options, the
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
python /path/to/tests/python/benchmarks/350ModelBinningResponseBenchmark.py \
  --events 100000 --bins 4 64 --threads 1 2 4 8 --repeat 7
```

Pass `--width-fraction 0.25` to benchmark relative widths. Without this option,
the benchmark uses the absolute-width path.

Run it in the test Python environment with the intended compiled GUNDAM module on
`PYTHONPATH`. The JSON output includes the loaded module path, median timings,
physical-bin and outside contents and MC errors. Compare builds with the same
inputs, compiler options and an otherwise idle machine; there is no timing threshold in the test suite.

## PlotGenerator

[< back to parent (Propagator)](#propagator-configuration)

### Description

### Config options
Note that the leaf used in the `varDictionnaries` must be an int.

### Example

```yaml
varDictionnaries:
  - name: "reaction"
    dictionary:
      - { value: -1, title: "no truth", color: 1 } # kBlack
      - { value: 0, title: "CCQE", color: 801 } # kOrange+1
      - { value: 1, title: "RES", color: 413 } # kGreen-3
      - { value: 2, title: "DIS", color: 843 } # kTeal+3
      - { value: 3, title: "COH", color: 867 } # kAzure+7
      - { value: 4, title: "NC", color: 430 } # kCyan-2
      - { value: 5, title: "CC-#bar{#nu}_{#mu}", color: 593 } # kBlue-7
      - { value: 6, title: "CC-#nu_{e}, CC-#bar{#nu}_{e}", color: 602 } # kBlue+2
      - { value: 7, title: "out FV", color: 909 } # kPink+9
      - { value: 9, title: "2p2h", color: 634 } # kRed+2
      - { value: 777, title: "sand #mu", color: 920 } # kGray
      - { value: 999, title: "other", color: 922 } # kGray+2

histogramsDefinition:
  - varToPlot: "Raw"

  - varToPlot: "Pmu"
    splitVars: ["", "reaction"]
    useSampleBinning: true # if not possible, error
    rescaleAsBinWidth: true # default true -> to look like a PDF
    rescaleBinFactor: 30 
    xMax: 3000.
    xTitle: "p_{#mu} (MeV)"
    yTitle: "Counts"

  - varToPlot: "CosThetamu"
    splitVars: ["", "reaction"]
    useSampleBinning: true # if not possible, error
    rescaleAsBinWidth: true # default true -> to look like a PDF
    rescaleBinFactor: 1 
    xMax: 1.
    xTitle: "cos#theta_{#mu}"
    yTitle: "Counts"

canvasParameters:
  height: 800
  width: 1200
  nbXplots: 3
  nbYplots: 2
```

## EventTreeWriter

[< back to parent (Propagator)](#propagator-configuration)

### Description


### Config options

| Option                                         | Type | Description                                                         | Default |
|------------------------------------------------|------|---------------------------------------------------------------------|---------|
| writeDials                                     | bool | If true, then save a TGraph for each dial in each even              | false   |
| nPointsPerDial                                 | int  | The number of points to use between +5 and -5 sigma                 | 3       |

## Parameter Scanner Config

[< back to parent (FitterEngine)](#fitterengineconfig)

### Scan options

| scanConfig Options  | Type         | Description                                         | Default |
|---------------------|--------------|-----------------------------------------------------|---------|
| nbPoints            | int          | N sample point of LLH scan                          | 100     |
| parameterSigmaRange | list(double) | Scan around the current point at +/- X prior sigmas | {-3, 3} |
| varsConfig          | json         | List of quantities to scan                          |         |
| useParameterLimits  | bool         | Don't scan LLH out of bounds                        | true    |


### Scan/Vars options

| varsConfig Options     | Type | Description                                | Default |
|------------------------|------|--------------------------------------------|---------|
| llh                    | bool | Save total LLH scans                       | true    |
| llhStat                | bool | Save stat. LLH scans                       | true    |
| llhPenalty             | bool | Save penalty LLH scans                     | true    |
| llhStatPerSample       | bool | Save stat. LLH per sample scans            | false   |
| llhStatPerSamplePerBin | bool | Save stat. LLH per sample per bin scans    | false   |
| weightPerSample        | bool | Save total weight per sample scans         | false   |
| weightPerSamplePerBin  | bool | Save total weight per sample per bin scans | false   |
