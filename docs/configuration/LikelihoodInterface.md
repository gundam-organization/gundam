## LikelihoodInterfaceConfig
[< back to parent (fitterEngineConfig)](./FitterEngine.md)

### Description

Define how the likelihood will be calculated.  This includes defining the
datasets that will be used for the model and measurement.  It also includes
defining the "propagator" that will apply the effect of fitting parameters
to the model.  The comparison between the measurement and model is defined
using `jointProbabilityConfig`.

### Config options

| Likelihood Interface Options | Type | Descriptions                                | default   |
|------------------------------|------|---------------------------------------------|-----------|
| propagatorConfig             | Json | Configure the parameter propagation         |           |
| dataSetList                  | Json | Configure the input data sets               |           |
| jointProbabilityConfig       | Json | Configure the joint probability calculation | BarlowLLH |
| plotGeneratorConfig          | Json |                                             |           |
| enableStatThrowInToys        | bool |                                             |           |
| gaussStatThrowInToys         | bool |                                             |           |
| enableEventMcThrow           | bool |                                             |           |

