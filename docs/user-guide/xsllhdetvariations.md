## xsllhDetVariations

This program performs the calculation of the detector covariance for HighLAND2 files. The `all_syst` tree must be enabled in HighLAND2 along with the systematic variations to be included in the covariance matrix.

It is designed to be run with a JSON configuration file. The usage of the command is as follows:
```bash
$ xsllhDetVar -j </path/to/config.json>
```
The `-j` flag is required and is the config file for the fit. Currently there are no other options; all settings are specified in the configure file.

### Configuration file

Examples in the repository. Full description coming soon.

