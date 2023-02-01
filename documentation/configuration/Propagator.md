## Propagator

[< back to parent (FitterEngine)](./FitterEngine.md)

### Description


### Config options

| Option                                         | Type | Description                                                         | Default |
|------------------------------------------------|------|---------------------------------------------------------------------|---------|
| [fitSampleSetConfig](./FitSampleSet.md)        | json | FitSampleSet config                                                 |         |
| [parameterSetListConfig](./FitParameterSet.md) | json | ParameterSetList config                                             |         |
| [dataSetList](./DatasetLoader.md)              | json | DatasetManager config                                               |         |
| [plotGeneratorConfig](./PlotGenerator.md)      | json | PlotGenerator config                                                |         |
| showEventBreakdown                             | bool | Print sample total weight                                           | true    |
| enableStatThrowInToys                          | bool | Throw statistical error with a poisson distribution                 | true    |
| enableEventMcThrow                             | bool | Each MC event get reweighted with Poisson(1)                        | true    |
| throwAsimovFitParameters                       | bool | Throw parameters of MC before fit (used to test fitter convergence) | false   |
| reThrowParSetIfOutOfBounds                     | bool | If any thrown parameter of the set is out of bounds, throw again    | true    |

