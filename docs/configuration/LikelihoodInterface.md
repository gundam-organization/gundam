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
| dataSetList                         | Json | Configure the input data sets               |           |
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
