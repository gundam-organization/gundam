## Propagator Configuration

[< back to parent (likelihoodInterfaceConfig)](LikelihoodInterface.md)

### Description

The propagator handles the engine in charge of applying parameters to the
samples.  It owns the parameters, the samples, and the data loader.

### Config options

| Option                                                      | Type   | Description                                                                                | Default |
|-------------------------------------------------------------|--------|--------------------------------------------------------------------------------------------|---------|
| [sampleSetConfig](SampleSet.md)                             | json   | Configure the samples that will be compared data vs model                                  |         |
| [parametersManagerConfig/parameterSetList](ParameterSet.md) | json   | Configure the parameters that will affect the model                                        |         |
| [parameterInjection/parameterSetList](ParameterSet.md)      | Json   | Override values for the parameters                                                         |         |
| showNbEventParameterBreakdown                               | bool   | Print the number of event affected by each parameter                                       | false   |
| showNbEventPerSampleParameterBreakdown                      | bool   | Print the number of event per sample affected by each parameter                            | false   |
| globalEventReweightCap                                      | double | Will cap the weight applied by the parameters: evWeight = baseWeight * min(parWeight, cap) | nan     |
| debugPrintLoadedEvents                                      | bool   | If true then print contents for a set of events.                                           | false   |
| debugPrintLoadedEventsNbPerSample                           | int    | Number of events to dump in the debugPrintLoadedEvents output                              | 5       |
| devSingleThreadReweight                                     | bool   | Use a single thread for reweighting events with the CPU                                    | false   |
| devSingleThreadHistFill                                     | bool   | Use a single thread for filling the histograms with the CPU                                | false   |


