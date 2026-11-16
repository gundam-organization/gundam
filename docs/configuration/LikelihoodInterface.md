## LikelihoodInterfaceConfig
[< back to parent (fitterEngineConfig)](./FitterEngine.md)

### Description

Define how the likelihood will be calculated.  This includes defining the
datasets that will be used for the model and measurement.  It also includes
defining the "propagator" that will apply the effect of fitting parameters
to the model.  The comparison between the measurement and model is defined
using `jointProbabilityConfig`.

### Config options

| Likelihood Interface Options        | Type | Descriptions                                | default   |
|-------------------------------------|------|---------------------------------------------|-----------|
| [propagatorConfig](./Propagator.md) | Json | Configure the parameter propagation         |           |
| [datasetManagerConfig](./DatasetManager.md) | Json | Configure the input datasets | |
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
