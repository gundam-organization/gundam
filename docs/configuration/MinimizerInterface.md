## configMinimizerInterface

[< back to parent (FitterEngine)](FitterEngine.md)

The minimizer interface can be configured to use either an MCMC to estimate
the Bayesian posterior, or a maximum likelihood optimization to find
frequentist estimators for the parameters (i.e. the maximum likelihood best
fit point), and the covering region (i.e. the Covariance at the best fit
point).  The maximum likelihood optimization is controlled using the
RootMinimizer object, while the MCMC is produced using the SimpleMcmc
object.  Both objects are configured using the configMinimizerInterface,
and are described separately.

### Config options for both minimizers

| minimizerConfig Options         | Type   | Description                                                            | Default       |
|---------------------------------|--------|------------------------------------------------------------------------|---------------|
| type                            | string | Choose the SimpleMcmc or RootMinimizer                                 | RootMinimizer |
| checkParameterValidity          | bool   | Turn on parameter validity checks (in physical, in domain, etc)        | false         |
| showParametersOnFitMonitor      | bool   | Display fit parameter parameter values on the monitor                  | false         |
| enablePostFitErrorFit           | bool   | Apply error evaluation after the minimization (may not be implemented) | true          |
| maxNbParametersPerLineOnMonitor | int    | Number of parameters on a monitor line                                 | 15            |
| useNormalizedFitSpace           | bool   | use fit parameter interface to provide prior mean at 0 and stddev at 1 | true          |
| writeLlhHistory                 | bool   | write a ttree registering all the llh evaluations                      | false         |

The rest of the fields are specific to the specific minimizer type being used.

- The [ROOT Minimizer](RootMinimizerConfig.md) provides a maximum likelihood optimization to find the best fit point of the likelihood and provides ways to calculate the Hessian at the best fit point.

- The [Simple MCMC](SimpleMcmcConfig.md) generates an MCMC chain of points drawn from the likelihood that characterizes the multi-dimension likelihood distribution (mostly used in Bayesian analysis).

