## MinimizerInterface

[< back to parent (FitterEngine)](./FitterEngine.md)

### Config options

| minimizerConfig Options        | Type   | Description                                                                  | Default             |
|--------------------------------|--------|------------------------------------------------------------------------------|---------------------|
| minimizer                      | string | [engine name](https://root.cern.ch/doc/master/NumericalMinimization_8C.html) | Minuit2             |
| algorithm                      | string | algorithm name                                                               | default from engine |
| useNormalizedFitSpace          | bool   | use fit parameter interface to provide prior mean at 0 and stddev at 1       | true                |
| errorAlgo / errors             | string | algorithm to run after the fit (HESSE or MINOS)                              | Hesse               |
| enablePostFitErrorFit          | bool   | enable errorAlgo after fit has succeeded                                     | true                |
| tolerance                      | double | defines the required Estimated Distance from the Minimum stopping the fit    | 1E-4                |
| maxFcnCalls / max_fcn          | int    | maximum number of function calls before stopping fit                         | 1E9                 |
| maxIterations / max_iter       | int    | maximum number of minimizer iterations before stopping fit                   | 500                 |
| strategy                       | int    | [fitter strategy (sec. 1.3)](https://root.cern.ch/download/minuit.pdf)       | 1                   |
| print_level                    | int    | [minimizer verbose level (p.23)](https://root.cern.ch/download/minuit.pdf)   | 1                   |
| enableSimplexBeforeMinimize    | bool   | run SIMPLEX before the real fit (can help to find the minimum)               | false               |
| simplexMaxFcnCalls             | int    | stop SIMPLEX after N calls                                                   | 1000                |
| simplexToleranceLoose          | int    | loosing up minimizer by this factor                                          | 1000                |
| simplexStrategy                | int    | strategy for SIMPLEX                                                         | 1                   |
| generatedPostFitParBreakdown   | bool   | Generate figures showing hessian eigen decomp breakdown by parameter         | false               |
| generatedPostFitEigenBreakdown | bool   | Generate figures showing parameter breakdown by hessian eigen                | false               |
| monitorRefreshRateInMs         | int    | Max refresh rate for the fit monitor in milliseconds                         | 5000                |
| showParametersOnFitMonitor     | bool   | Display fit parameter parameter values on the monitor                        | false               |
