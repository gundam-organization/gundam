## Binning files

The Super-xsllh framework uses a simple implementation of a binning file to define the binning for various parts of the code as plain text files. The binning file contains two pieces of information, the number of dimensions used in the binning (e.g. 1D, 2D, etc. binning) on the first line of the file, and a line for each bin defined using the upper and lower edges. The bins inside the file must be arranged from lowest to highest starting with the last dimension, then each previous dimension.

When determining which bin an event falls into for the given variable(s), the bin spans the range \(L \leq x < U\) where \(L,U\) are the lower and upper limits of the bin. The only important whitespace is the space between numbers and each bin must be on a separate line.

An example one dimensional binning file:
```nohighlight
1
-1.0  0.2
 0.2  0.6
 0.6  0.7
 0.7  0.8
 0.8  0.85
 0.85 0.90
 0.90 0.94
 0.94 0.98
 0.98 1.00
```
This defines nine cosine bins using their lower and upper edges. The first line is for specifying the number of dimensions.

An example two dimensional binning file:
```nohighlight
2
-1.0  0.2  000  30000
 0.2  0.6  000  300
 0.2  0.6  300  400
 0.2  0.6  400  500
 0.2  0.6  500  600
 0.2  0.6  600  30000
 0.6  0.7  000  300
 0.6  0.7  300  400
 0.6  0.7  400  500
 0.6  0.7  500  600
 0.6  0.7  600  800
 0.6  0.7  800  30000
 0.7  0.8  000  300
 0.7  0.8  300  400
 0.7  0.8  400  500
 0.7  0.8  500  600
 0.7  0.8  600  800
 0.7  0.8  800  30000
 0.8  0.85 000  300
 0.8  0.85 300  400
 0.8  0.85 400  500
 0.8  0.85 500  600
 0.8  0.85 600  800
 0.8  0.85 800  1000
 0.8  0.85 1000 30000
 0.85 0.9  000  300
 0.85 0.9  300  400
 0.85 0.9  400  500
 0.85 0.9  500  600
 0.85 0.9  600  800
 0.85 0.9  800  1000
 0.85 0.9  1000 1500
 0.85 0.9  1500 30000
 0.9  0.94 000  400
 0.9  0.94 400  500
 0.9  0.94 500  600
 0.9  0.94 600  800
 0.9  0.94 800  1250
 0.9  0.94 1250 2000
 0.9  0.94 2000 30000
 0.94 0.98 000  400
 0.94 0.98 400  500
 0.94 0.98 500  600
 0.94 0.98 600  800
 0.94 0.98 800  1000
 0.94 0.98 1000 1250
 0.94 0.98 1250 1500
 0.94 0.98 1500 2000
 0.94 0.98 2000 3000
 0.94 0.98 3000 30000
 0.98 1.00 000  500
 0.98 1.00 500  700
 0.98 1.00 700  900
 0.98 1.00 900  1250
 0.98 1.00 1250 2000
 0.98 1.00 2000 3000
 0.98 1.00 3000 5000
 0.98 1.00 5000 30000
```
This defines 58 bins with the cosine as the first dimension and momemtum as the second dimension.
