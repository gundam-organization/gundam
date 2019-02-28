## xsllhXsecCov

This program translates a text file containing the cross-section covariance matrix into a ROOT file for the fit.

It is designed to be run using a series of command line options. The usage of the command is as follows:
```bash
$ xsllhXsecCov
USAGE: xsllhXsecCov -i </path/to/cov.txt>
OPTIONS
-i : Input xsec file (.txt)
-o : Output ROOT filename
-m : Covariance matrix name
-b : Add value to diagonal
-r : Parameter mask
-C : Calculate correlation matrix
-I : Build INGRID covariance
-S : Store as TMatrixT
-h : Display this help message
```
The `-i` flag is required and is the cross-section covariance in a text file. The rest of the options are optional. The lowercase options take required parameters while the uppercase options are simple flags.
