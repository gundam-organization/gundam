## Root Minimizer Configuration

[< back to parent (MinimizerConfig)](MinimizerConfig.md)

The RootMinimizer uses ROOT to apply numeric minimization to the
likelihood and find the best fit point.

### Config options

These are the minimizerInterfaceConfig fields that are supported by when
the minimizer type is set to `RootMinimizer`.

| minimizerConfig Options         | Type   | Description                                                                  | Default             |
|---------------------------------|--------|------------------------------------------------------------------------------|---------------------|
| type                            | string | Must be "RootMinimizer" for this configuration                               |                     |
| checkParameterValidity          | bool   | Turn on parameter validity checks (in physical, in domain, etc)              | false               |
| showParametersOnFitMonitor      | bool   | Display fit parameter parameter values on the monitor                        | false               |
| enablePostFitErrorFit           | bool   | Evaluate the Hessian after the fit                                           | true                |
| maxNbParametersPerLineOnMonitor | int    | Number of parameters on a monitor line                                       | 15                  |
| useNormalizedFitSpace           | bool   | use fit parameter interface to provide prior mean at 0 and stddev at 1       | true                |
| writeLlhHistory                 | bool   | write a ttree registering all the llh evaluations                            | false               |
|---------------------------------|--------|------------------------------------------------------------------------------|---------------------|
| Root Minimizer Specific         | Type   | Description                                                                  | Default             |
|---------------------------------|--------|------------------------------------------------------------------------------|---------------------|
| minimizer                       | string | [engine name](https://root.cern.ch/doc/master/NumericalMinimization_8C.html) | Minuit2             |
| algorithm                       | string | algorithm name                                                               | default from engine |
| errors                          | string | algorithm to run after the fit (HESSE or MINOS)                              | Hesse               |
| tolerance                       | double | defines the required Estimated Distance from the Minimum stopping the fit    | 1E-4                |
| maxFcnCalls                     | int    | maximum number of function calls before stopping fit                         | 1E9                 |
| maxIterations                   | int    | maximum number of minimizer iterations before stopping fit                   | 500                 |
| strategy                        | int    | [fitter strategy (sec. 1.3)](https://root.cern.ch/download/minuit.pdf)       | 1                   |
| print_level                     | int    | [minimizer verbose level (p.23)](https://root.cern.ch/download/minuit.pdf)   | 1                   |
| enableSimplexBeforeMinimize     | bool   | run SIMPLEX before the real fit (can help to find the minimum)               | false               |
| enablePostFitErrorFit           | bool   | Evaluate the Hessian after the fit                                           | true                |
| simplexMaxFcnCalls              | int    | stop SIMPLEX after N calls                                                   | 1000                |
o| simplexToleranceLoose           | int    | loosing up minimizer by this factor during SIMPLEX                           | 1000                |
| simplexStrategy                 | int    | strategy for SIMPLEX                                                         | 1                   |
| generatedPostFitParBreakdown    | bool   | Generate figures showing hessian eigen decomp breakdown by parameter         | false               |
| generatedPostFitEigenBreakdown  | bool   | Generate figures showing parameter breakdown by hessian eigen                | false               |
| monitorRefreshRateInMs          | int    | Max refresh rate for the fit monitor in milliseconds                         | 5000                |
| monitorGradientDescent          | bool   |                                                                              |                     |
| tolerancePerDegreeOfFreedome    | int    |                                                                              |                     |
