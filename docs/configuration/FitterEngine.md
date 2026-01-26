## FitterEngineConfig

[< back to parent (gundamFitter)](../applications/gundamFitter.md)

### Description

### Config options

| FitterEngine Options                                | Type         | Description                                                                   | Default  |
|-----------------------------------------------------|--------------|-------------------------------------------------------------------------------|----------|
| [likelihoodInterfaceConfig](LikelihoodInterface.md) | json         | Configure the likelihood calculation.                                         | required |
| [minimizerConfig](MinimizerInterface.md)            | json         | MinimizerInterface config                                                     | required |
| [parameterScannerConfig](ParameterScanner.md)       | json         | Scan parameters config                                                        |          |
| scanConfig                                          | json         | Deprecated. See parameterScannerConfig.                                       |          |
| propagatorConfig                                    | json         | Deprecated. See likelihoodInterfaceConfig/propagatorConfig                    |          |
| mcmcConfig                                          | json         | Deprecated. See minimizerConfig                                               |          |
| engineType                                          | string       | Deprecated. See minimizerConfig/type                                          |          |
| enablePreFitScan                                    | bool         | Run fit parameter scan right before the minimization                          | false    |
| enablePostFitScan                                   | bool         | Run fit parameter scan right after the minimization                           | false    |
| enablePreFitToPostFitLineScan                       | bool         | Run scan between the prefit and postfit parameters                            | false    |
| generateSamplePlots                                 | bool         | Draw sample histograms according to the PlotGenerator config                  | true     |
| allParamVariations                                  | list(double) | List of points to perform individual parameter variation                      |          |
| enablePca / runPcaCheck / fixGhostFitParameters     | bool         | Dangerous: Fix parameter if the effect on stat LHH is lower than `pcaDeltaChi2Threshold` | false    |
| pcaThreshold                                        | double       | LLH threshold for PCA                                                         | 1E-6     |
| fixGhostEigenParametersAfterFirstRejected           | bool         | Fix all next parameters once PCA has been triggered (dev)                     | false    |
| throwMcBeforeFit                                    | bool         | Push MC parameter away from their prior before fitting                        | false    |
| throwMcBeforeFitGain                                | int          | Scale throws for MC parameters                                                | 1        |
| customFitParThrow*                                  | list         | Use the custom thrown values for parameters (dev)                             |          |
| scaleParStepWithChi2Response                        | bool         | Use LLH profile to scale parameter step size (dev)                            | false    |
| parStepGain                                         | bool         | Boost step value with `scaleParStepWithChi2Response` (dev)                    | 0.1      |
| savePostfitEventTrees                               | bool         | Save event tree after the fit.                                                | false    |

### JSON sub-structures

#### customFitParThrow options

| Options         | Type | Description                           | Default |
|-----------------|------|---------------------------------------|---------|
| **parIndex**    | int  | Parameter index to move               |         |
| **nbSigmaAway** | int  | Custom throw value (in unit of sigma) |         |
