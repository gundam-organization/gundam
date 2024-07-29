## FitterEngine Class Configuration

[< back to parent (gundamFitter)](../applications/gundamFitter.md)

### Description

Provide overall configuration for the FitterEngine class.  This is configured from YAML `fitterEngineConfig:`.

### fitterEngineConfig options

| fitterEngineConfig Options                            | Type         | Description                                                                   | Default |
|-------------------------------------------------------|--------------|-------------------------------------------------------------------------------|---------|
| [minimizerConfig](./MinimizerInterface.md)            | json         | MinimizerInterface config                                                     |         |
| [likelihoodInterfaceConfig](./LikelihoodInterface.md) | json         | LikelihoodInterface config                                                    |         |
| [parameterScannerConfig](./ParScanner.md)             | json         | ParameterScanner config                                                       |         |
| enablePreFitScan                                      | bool         | Parameter scan right before the minimization                                  | false   |
| enablePostFitScan                                     | bool         | Parameter scan right after the minimization                                   | false   |
| generateSamplePlots                                   | bool         | Draw sample histograms according to the PlotGenerator config                  | true    |
| enablePca                                             | bool         | Fix parameter if the effect on stat LHH is lower than `pcaDeltaChi2Threshold` | false   |
| pcaDeltaLlhThreshold                                  | double       | LLH threshold for PCA                                                         | 1E-6    |
| restoreStepSizeBeforeHesse                            | bool         | Use back original step size for error calculation                             | false   |
| enablePreFitToPostFitLineScan                         | bool         | Scan LLH along line between pre-fit and post-fit points                       | false   |
| generateSamplePlots                                   | bool         |                                                                               |         |
| generateOneSigmaPlots                                 | bool         |                                                                               |         |
| enableParameterVariations                             | bool         |                                                                               |         |
| allParamVariations                                    | list(double) | List of points to perform individual parameter variation                      |         |
| paramVariationSigmas                                  | list(double) | See allParamVariations                                                        |         |
| scaleParStepWithChi2Response                          |              |                                                                               |         |
| parStepGain                                           |              |                                                                               |         |
| throwMcBeforeFit                                      |              |                                                                               |         |
| throwMcBeforeFitGain                                  |              |                                                                               |         |
| savePostfitEventTrees                                 |              |                                                                               |         |


| Deprecated fitterEngineConfig Options | Type   | Description                                      | Default   |
|---------------------------------------|--------|--------------------------------------------------|-----------|
| [propagatorConfig](./Propagator.md)   | json   | See datasetManagerConfig/propagatorConfig        |           |
| [mcmcConfig](./MCMCInterface.md)      | json   | MinimizerInterface config                        |           |
| engineType                            | string | See minimizerConfig/type ("minimizer" or "mcmc") | minimizer |
| monitorRefreshRateInMs                | int    | See `minimizerConfig`                            |           |
| propagatorConfig                      |        |                                                  |           |
| scanConfig                            |        | See parameterScannerConfig                       |           |
| runPcaCheck                           | bool   | See enablePca                                    | false     |
| fixGhostFitParameters                 | bool   | See enablePca                                    | false     |
| fixGhostFitParameters                 | bool   | See enablePCA                                    |           |
| pcaDeltaChi2Threshold                 | double | See pcaDeltaLlhThreshold                         |           |
| ghostParameterDeltaChi2Threshold      | double | See pcaDeltaLlhThreshold                         |           |


| fitterEngineConfig Developer Options      | Type | Description                                                  | Default |
|-------------------------------------------|------|--------------------------------------------------------------|---------|
| fixGhostEigenParametersAfterFirstRejected | bool | Fix all next parameters once PCA has been triggered          | false   |
| throwMcBeforeFit                          | bool | Push MC parameter away from their prior before fitting       | false   |
| throwMcBeforeFitGain                      | int  | Scale throws for MC parameters                               | 1       |
| customFitParThrow*                        | list | Use the custom thrown values for parameters                  |         |
| scaleParStepWithChi2Response              | bool | Use LLH profile to scale parameter step size                 | false   |
| parStepGain                               | bool | Boost step value with `scaleParStepWithChi2Response`         | 0.1     |
| debugPrintLoadedEvents                    | bool | Printout `_debugPrintLoadedEventsNbPerSample_` loaded events | false   |
| debugPrintLoadedEventsNbPerSample         | int  | Number of event to print for each sample                     | 10      |

### JSON sub-structures

#### customFitParThrow options

| Options         | Type | Description                           | Default |
|-----------------|------|---------------------------------------|---------|
| **parIndex**    | int  | Parameter index to move               |         |
| **nbSigmaAway** | int  | Custom throw value (in unit of sigma) |         |
