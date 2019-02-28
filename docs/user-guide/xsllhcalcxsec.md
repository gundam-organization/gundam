## xsllhCalcXsec

This program performs the error propgation and cross-section calculation using the post-fit file from `xsllhFit`.

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

Examples in the repository. Full description coming soon.

