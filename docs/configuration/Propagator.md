## Propagator

[< back to parent (FitterEngine)](FitterEngine.md)

### Description

The propagator handles the engine in charge of applying parameters to the samples.
It owns the parameters, the samples, and the data loader.


### Config options

| Option                                      | Type   | Description                                                                                | Default |
|---------------------------------------------|--------|--------------------------------------------------------------------------------------------|---------|
| [fitSampleSetConfig](./SampleSet.md)        | json   | SampleSet config                                                                           |         |
| [parameterSetListConfig](./ParameterSet.md) | json   | ParameterSetList config                                                                    |         |
| [dataSetList](./DatasetDefinition.md)       | json   | DatasetManager config                                                                      |         |
| [plotGeneratorConfig](PlotGenerator.md)     | json   | PlotGenerator config                                                                       |         |
| [eventTreeWriter](EventTreeWriter.md)       | json   | EventTreeWriter config                                                                     |         |
| showEventBreakdown                          | bool   | Print sample total weight                                                                  | true    |
| showNbEventParameterBreakdown               | bool   | Print the number of event affected by each parameter                                       | false   |
| showNbEventPerSampleParameterBreakdown      | bool   | Print the number of event per sample affected by each parameter                            | false   |
| enableStatThrowInToys                       | bool   | Throw statistical error with a poisson distribution                                        | true    |
| enableEventMcThrow                          | bool   | Each MC event get reweighted with Poisson(1)                                               | true    |
| gaussStatThrowInToys                        | bool   | Throw statistical error with a gaussian distribution instead                               | false   |
| throwAsimovFitParameters                    | bool   | Throw parameters of MC before fit (used to test fitter convergence)                        | false   |
| globalEventReweightCap                      | double | Will cap the weight applied by the parameters: evWeight = baseWeight * min(parWeight, cap) | nan     |

