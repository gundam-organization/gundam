## Root Minimizer Configuration

[< back to parent (minimizerConfig)](MinimizerInterface.md)

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
| functionPrecision               | double | Relative numerical precision of the objective; must be finite and positive   | ROOT (double)       |
| maxFcnCalls                     | int    | maximum number of function calls before stopping fit                         | 1E9                 |
| maxIterations                   | int    | maximum number of minimizer iterations before stopping fit                   | 500                 |
| strategy                        | int    | [fitter strategy (sec. 1.3)](https://root.cern.ch/download/minuit.pdf)       | 1                   |
| print_level                     | int    | [minimizer verbose level (p.23)](https://root.cern.ch/download/minuit.pdf)   | 1                   |
| enableSimplexBeforeMinimize     | bool   | run SIMPLEX before the real fit (can help to find the minimum)               | false               |
| enablePostFitErrorFit           | bool   | Evaluate the Hessian after the fit                                           | true                |
| simplexMaxFcnCalls              | int    | stop SIMPLEX after N calls                                                   | 1000                |
| simplexToleranceLoose           | int    | loosen the minimizer tolerance by this factor during SIMPLEX                           | 1000                |
| simplexStrategy                 | int    | strategy for SIMPLEX                                                         | 1                   |
| generatedPostFitParBreakdown    | bool   | Generate figures showing hessian eigen decomp breakdown by parameter         | false               |
| generatedPostFitEigenBreakdown  | bool   | Generate figures showing parameter breakdown by hessian eigen                | false               |
| monitorRefreshRateInMs          | int    | Max refresh rate for the fit monitor in milliseconds                         | 5000                |
| monitorGradientDescent          | bool   |                                                                              |                     |
| tolerancePerDegreeOfFreedome    | int    |                                                                              |                     |

### Objective function precision

`functionPrecision` is passed to the numerical minimizer through `SetPrecision()`.
It describes the precision of likelihood evaluations and affects numerical
derivatives, including Hesse. It is distinct from `tolerance`, which controls
convergence through the estimated distance from the minimum.

When omitted, GUNDAM uses the active backend's recommendation if available.
MPS recommends float epsilon (`1.1920928955078125e-7`) because its event weights
and histogram outputs use float. CPU and reference propagation leave ROOT's
precision default unchanged. An explicit `functionPrecision` takes precedence.
The selected recommendation or override is printed during initialization; the
minimizer precision is also saved as `fit/minimizer/functionPrecision_TNamed`
under the fitter output directory.

A value smaller than the actual numerical noise can produce unreliable Hesse
errors. Increasing it allows larger differentiation steps, but does not restore
double precision to GPU calculations. Check uncertainty stability against CPU
and variations of this setting for the analysis being fitted. See the
[MPS Hesse investigation](../guides/MpsHessePrecision.md) for measured limits.
