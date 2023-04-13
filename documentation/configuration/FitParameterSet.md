## FitParameterSet

[< back to parent (Propagator)](./Propagator.md)

### Description

### Config options


| Option                                                 | Type       | Description                                                                                       | Default |
|--------------------------------------------------------|------------|---------------------------------------------------------------------------------------------------|---------|
| **name**                                               | string     | parameter set name                                                                                |         |
| isEnabled                                              | bool       | use in the propagator                                                                             | true    |
| [parameterDefinitions](./FitParameter.md)              | list(json) | config of individual parameter                                                                    |         |
| [dialSetDefinitions](./DialSet.md)                     | list(json) | config of dials                                                                                   |         |
| parameterDefinitionFilePath / covarianceMatrixFilePath | string     | path to root file containing cov matrix                                                           |         |
| covarianceMatrixTMatrixD                               | string     | path to covariance matrix in root file                                                            |         |
| parameterPriorTVectorD                                 | string     | path to list of parameter priors in root file                                                     |         |
| parameterNameTObjArray                                 | string     | path to list of parameter names in root file                                                      |         |
| parameterLowerBoundsTVectorD                           | string     | path to list of parameter min values in root file                                                 |         |
| parameterUpperBoundsTVectorD                           | string     | path to list of parameter max values in root file                                                 |         |
| enabledThrowToyParameters                              | bool       | throw parameters according to cov matrix if toy fit is selected                                   | true    |
| throwEnabledList                                       | string     | path to list of parameter throw states in root file                                               |         |
| useEigenDecompInFit                                    | bool       | eigen decompose the prior matrix                                                                  | false   |
| enablePca / fixGhostFitParameters                      | bool       | disable plot generation for the included parameters                                               | false   |
| parameterLimits*                                       | json       | global parameter limits definition                                                                | true    |
| numberOfParameters                                     | int        | manually specify the number of parameters to define<br/>(otherwise deduced for the cov matrix)    |         |
| nominalStepSize                                        | double     | define the prior scale of variation for the free parameters                                       |         |
| disableOneSigmaPlots                                   | bool       | disable plot generation for the included parameters                                               | false   |
| skipVariedEventRates                                   | bool       | disable event rate breakdown for the included parameters                                          | false   |
| useOnlyOneParameterPerEvent                            | bool       | at most one parameter from the set is attributed per event (faster data loading)                  | false   |
| useMarkGenerator                                       | bool       | Use Mark Hartz Cholesky decomposition implementation to throw correlated parameters               | false   |
| printDialSetsSummary                                   | bool       | print defined dialsets                                                                            | false   |
| maxNbEigenParameters (OLD)                             | int        | use only the N first eigen parameters with the highest eigen value                                | -1      |
| maxEigenFraction (OLD)                                 | double     | use only the N first eigen parameters which cover X % of the total variance (sum of eigen values) | 1.      |


### JSON sub-structures

#### parameterLimits options

| Options  | Type   | Description                          | Default |
|----------|--------|--------------------------------------|---------|
| minValue | double | min value of parameters from the set | nan     |
| maxValue | double | max value of parameters from the set | nan     |
