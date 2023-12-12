## SampleSet

[< back to parent (Propagator)](./Propagator.md)

### Description

### Config options

| Option                          | Type       | Description                                               | Default    |
|---------------------------------|------------|-----------------------------------------------------------|------------|
| [fitSampleList](./Sample.md) | list(json) | list of Sample config                                  |            |
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

