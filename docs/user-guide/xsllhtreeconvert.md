## xsllhTreeConvert

This program provides a base for translating event trees into the format expected by the fit code. This has basic support for HighLAND2 files, but still may require analysis specific tweaks. The "flattree" for the fit contains a small set of variables for both selected and true events.

It is designed to be run with a JSON configuration file. The usage of the command is as follows:
```bash
$ xsllhTreeConvert -j </path/to/config.json>
```
The `-j` flag is required and is the config file for the fit. Currently there are no other options; all settings are specified in the configure file.

### Output Format

The main fit code, `xsllhFit`, requires a specific format of input tree to read. The fit code expects a ROOT file with two event trees: `selectedEvents` and `trueEvents`. The structure of the trees is as follows:

The `selectedEvents` tree contains the selected events for the analysis. It contains a number of variables containing true information about the event, such as topology and reaction, along with the variables to perform the fit with. These are unimaginitively named `D1` and `D2`, and the fit requires the true and reconstructed values for each event.

+ **reaction** : [int] the true reaction for the event (e.g. CCQE, CCRES, NC, etc.)
+ **topology** : [int] the true topology for the event (e.g. CC0pi, CC1pi+, etc.)
+ **target** : [int] the true target in the interaction (e.g. carbon)
+ **nutype** : [int] the true neutrino type (e.g. numu)
+ **cut_branch** : [int] the sample ID for the event (e.g. muTPC)
+ **enu_true** : [float] true neutrino energy
+ **enu_reco** : [float] reconstructed neutrino energy
+ **q2_true** : [float] true Q2 for the event
+ **q2_reco** : [float] reconstructed Q2 for the event
+ **weight** : [float] the event weight
+ **D1True** : [float] true value for the first fit variable (e.g. muon momuntum)
+ **D1Reco** : [float] reco value for the first fit variable
+ **D2True** : [float] true value for the second fit variable (e.g. muon angle)
+ **D2Reco** : [float] reco value for the second fit variable

Some of these variables are flexible in their exact values, but they need to be consistant throughout the fit. For example, the reaction and topology can be mapped to any value (as long as it is an integer) for a given analysis. The values commonly used are the HighLAND codes for a given analysis. The fit is currently setup to fit in only two variables.

In the base example code \(Q^2\) is calculated for each event, but this is not strictly necessary unless the analysis needs it.

The `trueEvents` tree contains the true generated events for the analysis, including events that were not selected which is needed for the efficiency correction. It is almost the same format as the selected events tree, but without the reconstructed values for variables.

+ **reaction** : [int] the true reaction for the event (e.g. CCQE, CCRES, NC, etc.)
+ **topology** : [int] the true topology for the event (e.g. CC0pi, CC1pi+, etc.)
+ **target** : [int] the true target in the interaction (e.g. carbon)
+ **nutype** : [int] the true neutrino type (e.g. numu)
+ **cut_branch** : [int] the sample ID for the event
+ **enu_true** : [float] true neutrino energy
+ **q2_true** : [float] true Q2 for the event
+ **weight** : [float] the event weight
+ **D1True** : [float] true value for the first fit variable (e.g. muon momuntum)
+ **D2True** : [float] true value for the second fit variable (e.g. muon angle)

These variable definitions or mapping should match the selected events tree.

### Analysis Specific Modifications

It is very likely that a given analysis will need to modify the base tree converter, or even write a new one, to include or translate the necessary information required for the fit. The base code in the repository is setup to read HighLAND files and simply directly store the values from the HighLAND tree into the output tree.

Any form of tree converter can be used as long as it saves the correction information using the variable names above. If the output variable names are changed, then they also need to be changed in the main fit code. Change the code at your own risk.

### Configuration File

For using the base provided tree converter, this is a description of each option:

+ **output** : JSON object for output file options
    - **fname** : [string] name of the output file (e.g. test.root)
    - **sel_tree** : [string] name of the output selected events tree (should leave this as `selectedEvents`)
    - **tru_tree** : [string] name of the output true events tree (should leave this as `trueEvents`)
+ **highland_files** : JSON array of objects holding information for each HighLAND file to process
    - **fname** : [string] full path name of the HighLAND file to process
    - **file_id** : [int] user-defined file ID
    - **sel_tree** : [string] name of tree containing selected events (e.g. `default` for HL files)
    - **tru_tree** : [string] name of tree containing true events (e.g. `truth` for HL files)
    - **num_branches** : [int] number of branches in the selection
    - **cut_level** : [array] an array containing the `accum_level` cut values for the analysis in branch order
    - **samples** : [object] an object giving the mapping from branches to samples for the fit. The keys are the sample id's for the fit, and the values are arrays containing the branches to put in a given sample. For one-to-one mapping of branches to samples, use `{"0" : [0], "1" : [1], "2" : [2]}` for all the branches, in this case all three. To combine branches use `{"0" : [0,1], "1" : [2]}` which would combine branches zero and one to sample zero, and branch two to sample one for the fit.
    - **use** : [bool] If true, process HL file. If false, skip this file.
    - **sel_var** : JSON object containing the mapping of HL variables to fit variables for selected events
    - **tru_var** : JSON object containing the mapping of HL variables to fit variables for true events

The `highland_files` array can contain any number of HL files. An example configuration file can be found in the repository.

The `samples` mapping can be used to place events from multiple HL trees into the same sample ID for the fit, or separate HL files into different samples.
