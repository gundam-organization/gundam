## xsllhCalcXsec

This program performs the error propagation and cross-section calculation using the post-fit file from `xsllhFit`.

It is designed to be run with a JSON configuration file, and has CLI options which override certain options in the configure file. The usage of the command is as follows:
```
$ xsllhCalcXsec [-i,o,n,p,m,t,h] -j </path/to/config.json>
USAGE: xsllhCalcXsec
OPTIONS:
-j : JSON input
-i : Input file (overrides JSON config)
-o : Output file (overrides JSON config)
-n : Number of toys (overrides JSON config)
-p : Use prefit covariance for error bands
-m : Use mean of toys for covariance calculation
-t : Save toys in output file
-h : Print this usage guide
```
The `-j` flag is required and is the config file for the error propagation. The other flags are optional and take priority over the same options in the configure file if applicable.

### Configuration file

An example configuration file can be found in the repository. This is a description of each option.

- **input_dir** : [string] The input directory to find the input file, configure files, etc. This string is prepended to all file paths in the code, except the output file.
- **input_fit_file** : [string] The output file from `xsllhFit`.
- **output_file** : [string] The name of the output file.
- **extra_hists** : [string] [optional] The name of a text file specifying extra histograms from `input_fit_file` to save in the output.
- **num_toys** : [int] The number of toys to generate.
- **rng_seed** : [int] The seed to use for the random number generator. Setting this to zero will use a different seed for each run.
- **sel_config** : [string] The fit configuration file for the selected events.
- **tru_config** : [string] The fit configuration file for the true events.
- **decomposition** : Options for how the decomposition for toy throwing is performed.
    + **do_force_posdef** : [bool] If true, try to force the covariance matrix to be positive definite before giving up.
    + **force_posdef_val** : [double] The value to add to the diagonal each iteration to force positive definite.
    + **incomplete_chol** : [bool] Perform incomplete Cholesky decomposition.
    + **drop_tolerance** : [double] Threshold to drop entries from the covariance matrix when using incomplete Cholesky decomposition.
- **sig_norm** : Signal definitions and the signal specific normalizations. This is an object containing each signal definition by name, where the name must match the one from the fit configuration file.
    + **flux_file** : [string] The file containing the flux histogram.
    + **flux_hist** : [string] The name of the flux histogram in the file.
    + **flux_int** : [double] The flux integral for the flux normalization.
    + **flux_err** : [double] The error on the flux integral.
    + **use_flux_fit** : [bool] If true, use the fit parameters to reweight the flux histogram to calculate the flux integral. If false, use the numbers defined above to perform the flux normalization and error.
    + **num_targets_val** : [double] The number of targets normalization.
    + **num_targets_err** : [double] The error on the number of targets.
    + **relative_err** : [bool] If true, the numbers for the flux error and targets error are treated as relative errors. If false, they are treated as absolute errors.

### Overview

This code performs two main functions: calculating the cross-section, and propagating the post-fit errors to the final cross-section. The output from `xsllhFit` is an event spectrum, and needs to be properly normlized to calculate the final cross-section. In addition the output covariance matrix is the errors on the fit parameters themselves, which then needs to be propagated to the error on the cross-section.

The calculation from the number of selected events to cross-section requires several normalizations or corrections: the efficiency correction, flux normalization, bin-width normalization, and number of targets normalization.

The error propagation is performed by generating many correlated random throws of the fit parameters according to their post-fit error and calculating a cross-section for each throw. This generates many variations of the fit parameters, and many variations of the final calculated cross-section. The distribution of the cross-section throws is used to calculate a new covariance matrix, which gives the final error. A more in depth discussion of the math for this procedure can be found in [here](../reference/toythrows.md) in the reference section.

The cross-section for each signal definition is saved along with the covariance matrix which gives both the errors and correlations for each signal and between signals.

### Calculating Cross-sections

Calculating the cross-section is performed in three main steps: retreving selected signal events, applying efficiency correction, applying the rest of the normalizations. The `FitObj` is responsible for creating histograms filled with the selected signal events in the signal binning, which it returns through `GetSignalHist()`. The histograms are returned in a vector where each vector element corresponds to a different signal definition, and at this point are event distributions.

Next the efficiency correction is applied. The efficiency is calculated by taking the ratio between the selected signal events and the true generated signal events. Histograms for the true signal events are generated in the same way as the selected signal events, and selected and true events have separate instances of the `FitObj`. The `ApplyEff()` function takes the vectors of selected signal events and true generated signal events and takes the ratio of selected / true for each signal definition. This ratio is then applied bin-by-bin for each signal defintion. The histogram vectors are passed by reference and the correction is applied in place.

Then the remaining normalizations are applied at the same time through `ApplyNorm()`, which takes only the vector of selected signal histograms and a vector of parameter values corresponding to either the post-fit parameters or the current toy throw. For each signal definition, the number of targets normalization, the flux normalization, and bin-width normalizations are applied in order.

The number of target normalization simply divides the histogram by the number of targets specified in the configuration file or the number of targets for the given throw.

The flux normalization divides the histogram by the integral of the flux histogram specified in the configuration file. This can be done in two different ways: reweighting the flux histogram with the fit parameters, or by varying the flux integral specified in the configuration file. If using the fit parameters, the flux histogram is reweighted according to the flux parameters and then the integral is calculated from the reweighted flux.

The bin-width normalization divides each bin by its bin-width. The base unit of energy and momentum is MeV, so the bin-width normalization includes a parameter to use a different unit scaling to produce cross-sections normalized by GeV instead of MeV. The bin-width correction does not change when making toy throws.

Once the corrections and normalizations have been applied, the cross-sections have been fully calculated and they are stored in two different ways. The vector of histograms is stored, and a single histogram containing each signal concatenated together is also stored. When throwing toys, only the concatenated histogram is saved. The concatenated histogram is needed for calculating the covariance matrix.

### Propagating Errors

The error propagation is performed in two steps: generating many toy throws of the cross-section, and using those toys to calculate a covariance matrix. The `XsecCalc` class is responsible for generating the new parameter toy vectors and calculating the covariance matrix. The `FitObj` performs the event reweighting for a given toy parameter set, and the `ToyThrower` class performs the math to generate a new toy.

For each toy throw, a new vector is created and used to store the random vector from `ToyThrower::Throw()`, which is a vector containing a correlated throw of the parameter errors. The post-fit parameter values are then added to this error vector to create a new parameter vector where each parameter has been moved away from its post-fit value according to the post-fit covariance distribution.

This toy parameter vector is then given to the `FitObj` through `XsecCalc::ReweightEvents()` to weight the events according to the new parameters. Then the same procedure to calculate the cross-section is applied to the reweighted events. Finally the selected signal cross-section toy throw is stored as a single histogram containing each signal definition concatenated together.

Once all the toy throws are generated, they are used in `XsecCalc::CalcCovariance` to calculate the new covariance matrix for the cross-section using the analysis binning. This method can optionally use the post-fit cross-section or the mean of the toys to calculate the covariance matrix. Once the new covariance matrix is calculated, it is used to set the errors for each bin of the cross-section for each signal definition. Additionally the correlation matrix is also calculated and saved.

