## xsllhDetVariations

This program performs the calculation of the detector covariance for HighLAND2 files. The `all_syst` tree must be enabled in HighLAND2 along with the systematic variations to be included in the covariance matrix.

It is designed to be run with a JSON configuration file. The usage of the command is as follows:
```bash
$ xsllhDetVar -j </path/to/config.json>
```
The `-j` flag is required and is the config file for the fit. Currently there are no other options; all settings are specified in the configure file.

### Configuration file

An example configuration file can be found in the repository. This is a description of each option.

+ **fname_output** : [string] Name of the output file.
+ **covariance** : [bool] If true, calculate covariance matrix.
+ **covariance_name** : [string] Name of covariance matrix in the ROOT file.
+ **correlation_name** : [string] Name of correlation matrix in the ROOT file.
+ **weight_cut** : [float] Threshold to exclude events based on the event weight.
+ **single_syst** : [bool] If true, use only a single systematic from the file.
+ **syst_idx** : [int] The index of the single systematic to run.
+ **var_names** : [array] An array of strings defining the tree variables to use for the covariance matrix. For example: `["selmu_costheta", "selmu_mom"]`.
+ **projection** : [bool] Instead of saving plots with the axis as bin number, plot using a kinematic variable.
+ **plot_variable** : [string] Which variable from `var_names` to use when plotting if `projection` is enabled.
+ **pdf_print** : [bool] If true, print all plots to PDF and save in the current directory.
+ **cov_sample_binning** : [object] The definition of each covariance matrix sample and its binning. Further described below.
+ **files** : [array] The array of files to be processed. Further described below.
    - **fname_input** : [string] The path name to the HighLAND file.
    - **tree_name** : [string] Name of the TTree to read (e.g. `"all_syst"`).
    - **detector** : [string] Which detectors are in the file; this is a user defined name.
    - **num_toys** : [int] number of systematic toy throws in the file.
    - **num_syst** : [int] number of systematic parameters enabled.
    - **num_samples** : [int] number of samples in the selection.
    - **samples** : [object] Object defining the mapping of HighLAND branches to covariance samples.
    - **cuts** : [array] Array of numbers corresponding to the cut values for `accum_level` for each sample.
    - **use** : [bool] If true, use file in processing.

### Overview

The HighLAND files are loaded and the kinematic variables and event weights are read for each toy and systematic and used to build a covariance matrix. First a set of histograms to hold the toys and average are initialize based on the `cov_sample_binning` and the options for each file are read in from the configuration file. The files are looped over and all the toys in a given file are processed and the events are placed into the appropriate sample histogram. Finally the covariance and correlation matrix are calculated from the toys, and the distribution of the toys for each covariance sample are saved in the ROOT file.

The binning for each sample uses the `BinManager` class to process the binning files and perform the indexing. When defining the variables to use for the covariance matrix, the variables must be in the same order as the binning file.

The covariance matrix produced is the fractional covariance matrix, and thus represents the fractional detector error. It is calculated using:
$$
V_{ij} = \frac{1}{N_t} \sum^{N_t}_{t} (1 - b_{it}/\bar{b_i})(1 - b_{jt}/\bar{b_j})
$$

where \(V_{ij}\) is the \(ij^{\text{th}}\) covariance matrix element, \(N_t\) is the number of toys used, \(b_{it}\) is the value for bin \(i\) for toy \(t\), and \(\bar{b_i}\) is the average of the toys used for bin \(i\). If the files processed have differing number of toys, the lowest common denominator from all the files is used for the number of toys to calculate the covariance matrix.

### Covariance sample binning

The number of detector samples in the covariance matrix and their binning is defined by the `cov_sample_binning` option in the configuration file. The mapping between HighLAND selection branches and the numbering used when building the covairance matrix is defined by the user. The number of the covariance matrix sample is the number the fit will use when matching events to their matching detector covariance sample.

The sample binning is defined using a JSON Object where the keys correspond to the sample ID's and the values are the file path of the binning file to use for that particular sample. For example:
```json
"cov_sample_binning" : {
    "0" : "/path/to/sample0_binning.txt",
    "1" : "/path/to/sample1_binning.txt",
    "2" : "/path/to/sample2_binning.txt",
    "3" : "/path/to/sample3_binning.txt"
    }
```
which defines three samples each with their own binning file. Multiple samples can use the same binning file, each sample receives a copy of the file when initialized in the code. For a given analysis, Sample 0 may correspond to the muTPC sample while Sample 1 may correspond to the CC1pi control sample, etc.

### File sample mapping

The HighLAND selection branches in each file need to be mapped to covariance matrix samples, however not all branches need to be used. This is performed by using a JSON Object where the keys again correspond to the sample ID's defined in the covariance sample binning, and the values are arrays containing all the branch numbers to place in that given sample. For example:
```json
"files" : [
    {
        ...
        "samples" : {
            "0" : [0],
            "1" : [1],
            "2" : [2],
            "3" : [3]
            }
    }
]
```
which would map the HighLAND branches 0,1,2,3 to the same sample number. Multiple branches can be assigned to the same covariance matrix sample allowing for the combining of HighLAND branches like so:
```json
"files" : [
    {
        ...
        "samples" : {
            "0" : [0],
            "1" : [1],
            "2" : [2,4],
            "3" : [3,5,6]
            }
    }
]
```
which would assign both branches 2 and 4 to sample 2, and branches 3,5,6 to sample 3. The samples must match the same ones defined in `cov_sample_binning`, so in the previous examples trying to assign branches to a sample 4 would be incorrect.

Multiple files can assign their branches to the same samples as other files, or each sample can be independent by defining enough covariance matrix samples.
