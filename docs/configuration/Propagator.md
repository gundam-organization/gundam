## Propagator Configuration

[< back to parent (likelihoodInterfaceConfig)](LikelihoodInterface.md)

### Description

The propagator handles the engine in charge of applying parameters to the
samples.  It owns the parameters, the samples, and the data loader.

### Config options

| Option                                                      | Type       | Description                                                                                | Default |
|-------------------------------------------------------------|------------|--------------------------------------------------------------------------------------------|---------|
| [sampleSetConfig](SampleSet.md)                             | json       | Configure the samples that will be compared data vs model                                  |         |
| [parametersManagerConfig/parameterSetList](ParameterSet.md) | json       | Configure the parameters that will affect the model                                        |         |
| [parameterInjection/parameterSetList](ParameterSet.md)      | list(json) | Override values for the parameters                                                         |         |
| showNbEventParameterBreakdown                               | bool       | Print the number of event affected by each parameter                                       | false   |
| showNbEventPerSampleParameterBreakdown                      | bool       | Print the number of event per sample affected by each parameter                            | false   |
| globalEventReweightCap                                      | double     | Will cap the weight applied by the parameters: evWeight = baseWeight * min(parWeight, cap) | nan     |
| debugPrintLoadedEvents                                      | bool       | If true then print contents for a set of events.                                           | false   |
| debugPrintLoadedEventsNbPerSample                           | int        | Number of events to dump in the debugPrintLoadedEvents output                              | 5       |
| devSingleThreadReweight                                     | bool       | Use a single thread for reweighting events with the CPU                                    | false   |
| devSingleThreadHistFill                                     | bool       | Use a single thread for filling the histograms with the CPU                                | false   |

### CPU backend and event migration

With `likelihoodInterfaceConfig.backendManagerConfig` set to
`{isEnabled: true, type: CPU}`, the backend applies variable dials, updates bin
contributions and fills the histograms. Both `binningResponse` widths (`width`
and `widthFraction`) are supported, including events outside the binning domain.
Weight-only parameter changes reuse the cached contributions.

Without variable dials, dynamic rebinning remains explicit through
`Sample::updateEventBinning()` or `Propagator::updateEventSampleBinning()`.
With variable dials, explicitly dynamic samples also consume host variable edits
on each propagation. Automatically enabled dynamic samples retain their cached
contributions until a dependent input changes or rebinning is explicitly requested.

The CPU backend exposes `EventState` materialization for variables, bin indices
and contribution lists. Fit iterations can keep these intermediates in the
backend by disabling automatic materialization. The Python backend manager also
provides `setEnableAutoMaterialize()` and `materialize("EventState")` (alongside
`"EventWeights"` and `"Histograms"`). MPS migration and smoothed binning are not
implemented yet; requesting them reports an unsupported configuration.

A reproducible end-to-end benchmark is provided by
`tests/regular-tests/221CpuBinningBenchmark.py`.
