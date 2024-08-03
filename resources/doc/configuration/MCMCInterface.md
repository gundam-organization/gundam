## MCMCInterface

[< back to parent (FitterEngine)](FitterEngine.md)

### Config options

All of the options have usable default values.  The options marked "EX" are expert options and can usually be left at the default value.  The other options control basic running.

| mcmcConfig Options       | Type   | Description                                                                 | Default             |
|--------------------------|--------|-----------------------------------------------------------------------------|---------------------|
| algorithm | string | EX: The type of algorithm to use (only metropolis is supported) | "metropolis" |
| proposal | string | EX: The step proposal ("adaptive" or "simple") | "adaptive" |
| mcmcOutputTree | string | EX: The tree to save the mcmc steps into. | "MCMC" |
| saveRawSteps | bool | Drop the raw steps (in fitter space) if this is false | false |
| cycles | int | The number of cycles to run.  After debugging, set large enough to run a few hours. | 1 |
| steps | int | EX: The number of MCMC steps per cycle. (default is usually OK) | 10000 |
| likelihoodValidity | string | EX: Validity for parameters (def: "range,physical,mirror") | |
| modelSaveStride | int | Steps between saving the predicted sample histogram, zero to disable (for ppp). | 5000 |
| adaptiveRestore | string | Name of file to restore the state of the chain (set to "none" in config file | "none" |
| adaptiveCovFile | string | File with results of a MINUIT asimov fit.  Provides the step tuning. | "none" |
| adaptiveCovName | string | ROOT histogram name of a TH2D with HESS postfitCovariance | "see code" |
| adaptiveCovTrials | double | The number of effective trials to use for the input covariance | 500,000 |
| adaptiveCovWindow | int | The number of steps used to calculate the running proposal covariance | 1,000,000 |
| adaptiveCovDeweighting | double | EX: A detailed internal parameter.  See TSimpleMCMC | 0.0 |
| adaptiveFreezeCorrelations | int | Freeze the running covariance calculation after this many cycles | infinite |
| adaptiveFreezeAfter | int | Freeze the step length after this many cycles | infinite |
| adaptiveWindow | int | The number of steps used to estimate the acceptance | 1000 |
| burninCycles | int | Number of cycles to use in the burn-in phase (zero to disable) | 0 |
| burninSteps | int | EX: Number of steps per burn-in cycle | 10000 |
| burninResets | int | EX: Number of types to reset after burn-in cycle (zero to disable) | 0 |
| saveBurnin | bool | Save the burn-in steps (defaults to true, but should usually be false for production) | true |
| burninWindow | int | EX: Number of steps used to calculate the recent acceptance. | 1000 |
| burninCovWindow | int | EX: The number of steps in the running proposal covariance calculation | 1,000,000 |
| burninCovDeweighting | double | EX: A detailed internal parameter.  See TSimpleMCMC | 0.0 |
| burninFreezeAfter | int | Stop updating the proposal covariance after this many cycles | infinite |
