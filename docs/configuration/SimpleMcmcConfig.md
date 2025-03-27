## Simple MCMC Configuration

[< back to parent (MinimizerConfig)](MinimizerConfig.md)

The SimpleMcmc uses TSimpleMCMC to generate a MCMC that maps out the
likelihood.  The TSimpleMCMC templates implement several stepping
algorithms, but the main focus for GUNDAM is the "adaptive" proposal
algorithm.  This uses a metropolis-hasting acceptance, with a
multi-dimensional Gaussian proposal.  The covariance of the proposal is
updated by the accepted steps in the MCMC chain.  In general, the proposal
is allowed to evolve during the burn-in stage, and stops evolving while mapping
the posterior.

### Config options

These are the minimizerInterfaceConfig fields that are supported by when
the minimizer type is set to `SimpleMcmc`. All of the options have usable
default values.  The options marked "EX" are expert options and can usually
be left at the default value.  The other options control basic running.

| minimizerConfig Options         | Type   | Description                                                                           | Default      |
|---------------------------------|--------|---------------------------------------------------------------------------------------|--------------|
| type                            | string | Must be "RootMinimizer" for this configuration                                        |              |
| checkParameterValidity          | bool   | Turn on parameter validity checks (in physical, in domain, etc)                       | false        |
| showParametersOnFitMonitor      | bool   | Display fit parameter parameter values on the monitor                                 | false        |
| enablePostFitErrorFit           | bool   | Evaluate the Hessian after the fit                                                    | true         |
| maxNbParametersPerLineOnMonitor | int    | Number of parameters on a monitor line                                                | 15           |
| useNormalizedFitSpace           | bool   | use fit parameter interface to provide prior mean at 0 and stddev at 1                | true         |
| writeLlhHistory                 | bool   | write a ttree registering all the llh evaluations                                     | false        |
|---------------------------------|--------|---------------------------------------------------------------------------------------|--------------|
| mcmcConfig Options              | Type   | Description                                                                           | Default      |
|---------------------------------|--------|---------------------------------------------------------------------------------------|--------------|
| cycles                          | int    | The number of cycles to run.  After debugging, set large enough to run a few hours.   | 1            |
| steps                           | int    | The number of MCMC steps per cycle. (default is usually OK)                           | 10000        |
| algorithm                       | string | The type of algorithm to use (only metropolis is supported)                           | "metropolis" |
| proposal                        | string | EX: The step proposal ("adaptive" or "simple")                                        | "adaptive"   |
| mcmcOutputTree                  | string | EX: The tree to save the mcmc steps into.                                             | "MCMC"       |
| saveRawSteps                    | bool   | Drop the raw steps (in fitter space) if this is false                                 | false        |
| likelihoodValidity              | string | Control LLH parameter validity (def: "range,physical,mirror")                         |              |
| randomStart                     | bool   | Start from a random location                                                          | n/a          |
| sequence                        | string | See below                                                                             |              |
| burninSequence                  | string | See below                                                                             |              |
| modelSaveStride                 | int    | Steps between saving the predicted sample histogram, zero to disable (for ppp).       | 5000         |
| adaptiveRestore                 | string | Name of file to restore the state of the chain (set to "none" in config file          | "none"       |
| adaptiveCovFile                 | string | File with results of a MINUIT asimov fit.  Provides the step tuning.                  | "none"       |
| adaptiveCovName                 | string | ROOT histogram name of a TH2D with HESS postfitCovariance                             | "see code"   |
| adaptiveCovTrials               | double | The number of effective trials to use for the input covariance                        | 500,000      |
| adaptiveCovWindow               | int    | The number of steps used to calculate the running proposal covariance                 | 1,000,000    |
| adaptiveCovDeweighting          | double | A detailed internal parameter.  See TSimpleMCMC                                       | 0.0          |
| covarianceDeweighting           | double | A detailed internal parameter.  See TSimpleMCMC                                       | 0.0          |
| adaptiveFreezeCorrelations      | int    | Freeze the running covariance calculation after this many cycles                      | infinite     |
| adaptiveFreezeLength            | int    | Freeze the step length after this many cycles                                         | infinite     |
| adaptiveWindow                  | int    | The number of steps used to estimate the acceptance                                   | 1000         |
| burninCycles                    | int    | Number of cycles to use in the burn-in phase (zero to disable)                        | 0            |
| burninSteps                     | int    | EX: Number of steps per burn-in cycle                                                 | 10000        |
| saveBurnin                      | bool   | Save the burn-in steps (defaults to true, but should usually be false for production) | true         |
| burninWindow                    | int    | EX: Number of steps used to calculate the recent acceptance.                          | 1000         |
| burninCovWindow                 | int    | EX: The number of steps in the running proposal covariance calculation                | 1,000,000    |
| burninCovDeweighting            | double | EX: A detailed internal parameter.  See TSimpleMCMC                                   | 0.0          |
| saveRawSteps                    | bool   | Save the "internal transformation" steps.                                             |              |
|                                 |        |                                                                                       |              |

### Defining the MCMC sequence

The MCMC chain is generated in `cycles`, each of which as a fixed number of
`steps`.  The sequence of operation done during each cycle is defined by the
`sequence` (`burninSequence`) string, which provides a line that will be
executed by the ROOT C++ jit compiler.  The line can contain a single C++
statement (usually a loop).  The default value for the `sequence` is:
```C++
for (int chain = 0; chain < gMCMC.Cycles(); ++chain) {
    gMCMC.RunCycle("Chain", chain);
}
```
The default value for the `burninSequence` is:
```C++
for (int chain = 0; chain < gMCMC.Burning(); ++chain) {
      gMCMC.RunChain("Burn-in chain", chain);
}
```

The sequence can contain general C++ (to be interpreted by ROOT), as well
as several functions that are made available.  These change the parameters
of the MCMC stepper for the next cycle, and the parameters are restored to
the default values after each call to `RunCycle()`.  

- `gMCMC.RunCycle(std::string name, int id)` : Run a chain with the current
  configuration.  The `name` and `id` are used in the output statement, but
  have no other effect.
  
- `int gMCMC.Trials()` : Get the total number of steps that have been taken
  in the chain.  This includes steps taken during all previous cycles,
  including steps that may have been taken during previous jobs for this
  chain. 
      
- `int gMCMC.Cycles()` : Get the number of cycles to be run during this
  sequence.  This is set using the `cycles` field in the `minimizerConfig`.
  
- `int gMCMC.Steps()` : Get the number of steps in the current, or next,
  `RunCycle()`.  This takes a default value from the `steps` field in the
  `minimizerConfig`.  
  
- `gMCMC.Steps(int s)` : Set the number of steps for the next `RunCycle()`.

- `int Burnin()` : Get the number of cycles to be used for burn-in.

- `gMCMC.RandomStart(bool s)` : If true, the next `RunCycle()` will start
  from a random position.
  
- `gMCMC.SaveSteps()` : If true, the next `RunCycle()` will save output.
  (This should almost never be false).
  
- `gMCMC.ModelStride(int s)` : How frequently the model histogram should
  be saved to the output.  This is useful for the PPP calculation.
  
- `gMCMC.FreezeStep(bool b)` : The step length is frozen when this is
  true.  Otherwise, the step length will be adjusted to get the right
  acceptance rate.
  
- `gMCMC.FreezeCovariance(bool b)` : The proposal covariance is frozen when
  this is true.  Otherwise, the covariance is updated based on the accepted
  steps to get an estimate of the shape of the local posterior.

- `gMCMC.ResetCovariance()` : Reset the proposal covariance to the default
  value.
  
- `gMCMC.CovarianceWindow(int i)` : The number of points used to calculate
  the covariance of the posterior (This should typically be several times
  the length of a cycle). 
  
- `gMCMC.CovarianceDeweighting(double w)` : How much should a preexisting
  proposal covariance be deweighted when a new `RunCycle()` is started.  This should be
  between `0.0` and `1.0`.  A value of `0.0` means that the existing
  covariance should not be deweighted and be treated having
  `covarianceWindow` points.  A value of `1.0` means the existing
  covariance should be (almost) ignored. 
  
- `gMCMC.AcceptanceWindow(int i)` : The number of trials used to estimate
  the current acceptance.  This is estimating a binomial value, so it
  should typically have a value of a hundred to several hundred.  Very
  small and very large values will tend to make the step length unstable.
  
- `gMCMC.AcceptanceAlgorithm(int i)` : This should always be zero, unless
  you know what you are doing, and don't need the documentation.  That
  explicitly means, "read the code to understand what it does".
    
- `gMCMC.SetSigma(double s)` : Set the current step size in standard
  deviations of the current proposal covariance.
  
- `gMCMC.Restart()` : Restart the MCMC chain.  This is mostly useful when
  burning in problematic likelihoods (with "stupidly" bad correlations),
  and should be used with extreme care.  It should never be used while
  generating a chain that is intended to estimate the posterior.

#### Example burn-in sequences

The default sequences should probably be left alone, but it can be helpful
to change the burn-in sequence for problematic likelihoods (in practice, a
lot of real likelihoods are "problematic").  Here are some examples for
burn-in sequences that have been found to be useful and show the type of
control that is possible.

This is a burn-in sequence that can be useful when the starting point for
the chain may be very far from the "bulk" probability of the likelihood.
This allows the chain to relax into the right region of space, without
having the the final proposal be dominated by the initial walk.
```yaml
burninSequence: |
  for (int chain = 0; chain < gMCMC.Burnin(); ++chain) {
    gMCMC.AcceptanceWindow(100);
    gMCMC.CovarianceDeweighting(0.7);
    gMCMC.CovarianceWindow(10000+0.3*gMCMC.Burnin()*gMCMC.Steps());
    gMCMC.RunCycle("Burn-in",chain);
  }
```

This is the burn-in sequence that is used for standard MCMC validation
tests.  These tests are checking that the MCMC is properly handling cases
where parameters have physical bounds, and where the initial parameter
proposal covariance is a very poor description of the final posterior.
```yaml
burninSequence: |
  for (int chain = 0; chain < gMCMC.Burnin(); ++chain) {
    gMCMC.Steps(1000);
    gMCMC.AcceptanceWindow(100);
    gMCMC.CovarianceWindow(10000);
    gMCMC.CovarianceDeweighting(0.5);
    gMCMC.FreezeStep((chain > 2));
    gMCMC.FreezeCovariance(false);
    gMCMC.ResetCovariance((chain < 2));
    gMCMC.RunCycle("Burn-in",chain);
  }
```

This is the burn-in sequence that is used for the "Horrifying Likelihood"
validation test.  This test is looking at a multi dimensional likelihood
with two variables that are 100% correlated, and which has further
correlations between other variables.  In practice, if your likelihood is
this bad, you should redefine your model, but this is included to show what
can be done.
```yaml
burninSequence: |
  for (int chain = 0; chain < gMCMC.Burnin(); ++chain) {
    gMCMC.Restart();

    gMCMC.AcceptanceWindow(100);
    gMCMC.FreezeStep(chain > 0);
    gMCMC.FreezeCovariance(true);
    gMCMC.AcceptanceAlgorithm(1);
    gMCMC.Steps(2000);
    gMCMC.RunCycle("Downhill",chain);

    gMCMC.AcceptanceWindow(100);
    gMCMC.FreezeCovariance(true);
    gMCMC.Steps(2000);
    gMCMC.RunCycle("Wander",chain);

    gMCMC.Steps(1000);
    gMCMC.AcceptanceWindow(100);
    gMCMC.FreezeCovariance(false);
    gMCMC.RunCycle("Burn-in",chain);
  }
```
