## FitterEngine

[< back to parent (gundamFitter)](../applications/gundamFitter.md)

### Description

### Config options

| FitterEngine Options                                     | Type         | Description                                                                   | Default |
|----------------------------------------------------------|--------------|-------------------------------------------------------------------------------|---------|
| [propagatorConfig](./Propagator.md)                      | json         | Propagator config                                                             |         |
| [minimizerConfig](./MinimizerInterface.md)               | json         | MinimizerInterface config                                                     |         |
| [scanConfig](./ParScanner.md)                            | json         | Scan config                                                                   |         |
| enablePreFitScan                                         | bool         | Run fit parameter scan right before the minimization                          | false   |
| enablePostFitScan                                        | bool         | Run fit parameter scan right after the minimization                           | false   |
| generateSamplePlots                                      | bool         | Draw sample histograms according to the PlotGenerator config                  | true    |
| allParamVariations                                       | list(double) | List of points to perform individual parameter variation                      |         |
| enablePca / fixGhostFitParameters                        | bool         | Fix parameter if the effect on stat LHH is lower than `pcaDeltaChi2Threshold` | false   |
| pcaDeltaChi2Threshold / ghostParameterDeltaChi2Threshold | double       | LLH threshold for PCA                                                         | 1E-6    |
| fixGhostEigenParmetersAfterFirstRejected                 | bool         | Fix all next parameters once PCA has been triggered (dev)                     | false   |
| monitorRefreshRateInMs                                   | int          | Show fit stats every N milliseconds                                           | 500     |
| throwMcBeforeFit                                         | bool         | Push MC parameter away from their prior before fitting (dev)                  | false   |
| throwMcBeforeFitGain                                     | int          | Scale throws for MC parameters (dev)                                          | 1       |
| customFitParThrow*                                       | list         | Use the custom thrown values for parameters (dev)                             |         |
| scaleParStepWithChi2Response                             | bool         | Use LLH profile to scale parameter step size (dev)                            | false   |
| parStepGain                                              | bool         | Boost step value with `scaleParStepWithChi2Response` (dev)                    | 0.1     |
| restoreStepSizeBeforeHesse                               | bool         | Use back original step size for error calculation                             | false   |
| debugPrintLoadedEvents                                   | bool         | Printout `_debugPrintLoadedEventsNbPerSample_` loaded events  (dev)           | false   |
| debugPrintLoadedEventsNbPerSample                        | int          | Number of event to print for each sample (dev)                                | 10      |


### JSON sub-structures

#### customFitParThrow options

| Options         | Type | Description                           | Default |
|-----------------|------|---------------------------------------|---------|
| **parIndex**    | int  | Parameter index to move               |         |
| **nbSigmaAway** | int  | Custom throw value (in unit of sigma) |         |

