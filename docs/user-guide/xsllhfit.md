## xsllhFit

This program performs the minimization and fit to the data/MC producing a set of best-fit parameters and best-fit covariance matrix.

It is designed to be run with a JSON configuration file. The usage of the command is as follows:
```
$ xsllhFit [-n] [-h] -j </path/to/config.json>
```
The `-j` flag is required and is config file for the fit. The `-n` flag does a "dry-run" of the fit where it is initialized, but the minimization is not performed. The `-h` displays the help output and available options.

### Configuration file

Examples in the repository. Full description coming soon.
