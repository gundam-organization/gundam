## xsllhFit

This program performs the minimization and fit to the data/MC producing a set of best-fit parameters and best-fit covariance matrix.

It is designed to be run with a JSON configuration file. The usage of the command is as follows:
```bash
$ xsllhFit [-n,h] -j </path/to/config.json>
```
The `-j` flag is required and is config file for the fit. The `-n` flag does a "dry-run" of the fit where it is initialized but the minimization is not performed. The `-h` displays the help output and available options.

### Configuration file

Examples in the repository. Full description coming soon.

### Overview

This code initializes and performs the fit based on the configuration supplied. The primary output of `xsllhFit` is a set of post-fit values for each fit parameter and a post-fit covariance matrix describing the errors and correlations between fit parameters. Additional information is also saved: the chi-square per iteration is saved, post-fit correlation matrix, event distributions for each sample, histograms for the prior and final errors.

The code first loads the data and MC events and stores them in the specified samples according to the sample binning, and events are tagged as signal events (including which signal) and non-signal events (background events). The data events and histograms are never modified during the initialization or fit procedure. Then the nuisance parameters corresponding to flux, cross-section, and the detector parameters are initialized with their corresponding covariance matrices and supporting files. Finally, the minimizer is instantiated and initialized with the parameters and the minimization begins.

Many iterations of the minimization routine are performed as the best-fit values for each parameter are found according to the chi-square, and a second routine refines and estimates the errors on all the fit parameters. Finally the fit output information is collected and saved in the output ROOT file.

The fit as many configurable options which change how the fit is performed, making statistical and/or systematic throws of the parameters, performing principle component analysis, etc.

### Setting up the fit

Running the fit is mostly setup and building the inputs needed by the fit.

#### Defining Samples

Samples are collections of events corresponding some common crieteria or cuts. For example, defining a CC0pi sample where all the events are reconstructed as CC0pi events, or a sample based on the detectors like events with a muon and proton candidate in the TPC or the INGRID CC0pi sample.

Samples are defined by a unique ID number, which can be any positive integer (or zero). The events placed into the sample are chosen based on the `cut_branch` variable from the input fit flat tree. Additionally samples have a name (as a string) and a main detector they belong to which can be common to multiple samples.

Finally each sample needs a binning file to define the histogram bins. This binning file can be used for any number of samples.

Below is an example of the JSON to define a sample in the configuration file. Each sample definition is a JSON object in the sample array.
```json
"samples" : [
    {
        "cut_branch" : 0,
        "name" : "muTPC",
        "detector" : "ND280",
        "binning" : "binning/sample0_bins.txt",
        "use_sample" : true
    }
]
```
The `use_sample` option is a flag to enable/disable using the sample in the fit. If false, the sample is not read or used in the fit.

#### Defining Signals

Signal definitions are the criteria used to determine if a given event is a signal event regardless of sample placement. The signal definitions are defined separately for each detector in the fit.

Signals have a name (which needs to be unique), a binning file, and the variables and values to use for the definition. The signal definition is specified by using a JSON object which includes each variable and the values an event needs to match to be tagged as a signal event. The signal definitions needs to be mutually exclusive.

The binning file defines the bins used for the final cross-section for the signal and the template parameters in the fit. This is done so that samples can have binning independent of the signal definition.

Multiple variables and values can be used in a signal definition. An event is tagged as a signal event if it matches a value for every variable listed in the object, otherwise it is a background event.

Below is an example of the JSON to define two signals in the configuration file.
```json
"template_par" : [
    {
        "name" : "CC0pi_carbon",
        "binning" : "binning/analysis_binning.txt",
        "signal" : {"topology" : [0,1,2], "target" : [6]},
        "use" : true
    },
    {
        "name" : "CC1pi_carbon",
        "binning" : "binning/analysis_binning.txt",
        "signal" : {"topology" : [3], "target" : [6]},
        "use" : false
    }
],
```
This defines two signals using the topology and target variables. For an event to be tagged as a `CC0pi_carbon` event, it must have a topology of 0 or 1 or 2 and it must have a target value of 6. If the event has a topology of 3 and a target of 6, then it will be tagged as a `CC1pi_carbon` event. If it does not match any signal definition, it is a background event. 

Both signals use the same binning file, and in this example the `CC1pi_carbon` signal would not be used in the fit. The `use` option is a flag to enable/disable using the signal in the fit.

Inside the code, each signal is given an ID number according to the order in which they were defined. This ID number is used to label the signal for each event.
