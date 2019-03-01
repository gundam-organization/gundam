## xsllhXsecCov

This program translates a text file containing the cross-section covariance matrix into a ROOT file for the fit.

It is designed to be run using a series of command line options. The usage of the command is as follows:

```nohighlight
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

+ `-i` : Path to input text file.
+ `-o` : Output file name (default `xsec_cov.root`).
+ `-m` : Base name for matrix. `_cov` and `_cor` are appended for the covariance/correlation matrix respectively. Default name is `xsec`.
+ `-b` : Optional amount to add to the diagonal to force positive definite (e.g. `-b 0.0001`).
+ `-r` : Parameters not to include (explained below).
+ `-C` : Flag to calculate correlation matrix.

### Overview

A text file containing the covariance matrix elements as numbers is read in and saved as a `TMatrixTSym<double>` (by default) in the output ROOT file. The code also checks if the covariance matrix is invertible and can optionally remove parameters from the matrix automatically, instead of having to do this by hand in the original text file. If the matrix is not invertible, the code does not save anything in the output file.

*Important note*: this is the covariance matrix. The values on the diagonal are \(\sigma^2\).

### Example Text File

The first line of the text file **must** start with a number indicating the size of the matrix. Everything else on the first line is ignored, so it is suggested to list the parameter names to label the covariance matrix.

Below is an example of a text file using the FSI parameters. There are six parameters, so the first line starts with `6` and the parameter names are then ignored by the rest of the code. Each element must be separated by at least one space character, and must be arranged into columns and rows. The text file is read starting at the upper left element, and proceeds left to right, top to bottom. In this example, \(V_{00} = 0.1681\) and \(V_{55} = 0.2540\), with the indices matching how they would be stored in the computer.

```nohighlight
6 FSI_INEL_LO FSI_PI_ABS FSI_CEX_LO FSI_PI_PROD FSI_INEL_HI FSI_CEX_HI
0.1681   0.0248  0.0049 -0.0045  0.0000  0.0000
0.0248   0.2034 -0.0028 -0.0032  0.0000  0.0000
0.0049  -0.0028  0.3249 -0.0045  0.0000  0.0000
-0.0045 -0.0032 -0.0045  0.3745 -0.3026 -0.2916
0.0000   0.0000  0.0000 -0.3026  0.2500  0.2464
0.0000   0.0000  0.0000 -0.2916  0.2464  0.2540
```

### Masking Parameters

The input text file is not an easy file to edit once the number of parameters goes beyond three or four. If a parameter or multiple need to be removed for testing or other reasons, a parameter mask can be used. By specifying the `-r` flag along with a comma separated list of indices, the code will not include those parameters.

For example, if the above example text file was used as input with `-r 4,5` then `FSI_INEL_HI` and `FSI_CEX_HI` would not be included in the output covariance; the rows and columns containing those parameters would not be used. The input text file is not modified. The parameter mask does not need to have consecutive parameters, and the index starts at zero.
