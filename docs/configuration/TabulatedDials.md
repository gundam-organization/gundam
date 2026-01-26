# Tabulated Dials

A tabulated dial provides an efficient way to do an event-by-event reweight
of an event based on a truth variable and fitting parameters.  It is an
efficient way to handle a one-dimensional lookup table. For instance, this
can be used for long and short baseline oscillation fits that are dependent
on the true neutrino energy.  The table will be refilled for each set of
oscillation parameters.  An external library is used declare a look up
table, to fill the table for each iteration, and look up the position of
each event in the table based on truth variables.

The yaml for a Tabulated Dial is

```YAML
dialSetDefinitions:
  - dialType: Tabulated
    printDialsSummary: true
    dialInputList:
      - name: <fit-parameter1-name>
      - name: <fit-parameter2-name>
    tableConfig:
        - name: <table-name>                A table name passed to the
                                               library.
          bins:  <number>                   The (optional) expected number
                                               of bins in the table. Will
                                               be overridden by the
                                               initFunction.
          libraryPath: <path-to-library>    Location of the library
          initFunction: <init-func-name>    Initialization function name
          initArguments: [<arg1>, ...]      List of argument strings (e.g.
                                               input file names)
          updateFunction: <update-func-name> Function called to update table
          binningFunction: <bin-func-name>  Function to find bin index
          binningVariables: [<var1>, <var2>, ... ] Variables used for binning
                                               the table "X" coordinates
```

# Function to initialize the library

If `initFunction` is provided it will be called before calling the update or the binning functions.  It is called with the signature:

```
    extern "C"
    int initFunc(const char* name,
                 int argc, const char* argv[],
                 int bins);
```
* name -- The name of the table
* argc -- number of arguments
* argv -- argument strings.  The arguments are defined by the library, but are usually things like input file names for the lookup table information.
* bins -- The suggested size the table.  The library must choose an appropriate binning that will be used for the table.

The function should less than or equal to zero for failure, and otherwise, the number of elements needed to store the table (usually, the same as the input value of "bins").

# Library function to update the table

The update function is called before the events are reweighted.  The `updateFunction` signature is:

```
    extern "C"
    int updateFunc(const char* name,
                   double table[], int bins,
                   const double par[], int npar);
```
* name  -- table name
* table -- address of the table to update
* bins  -- The size of the table
* par   -- The parameters.  Must match parameters define in the dial definition
* npar  -- number of parameters

The function should return 0 for success, and any other value for failure

The table will be filled with "bins" values calculated with uniform spacing between "low" and "high".  If bins is one, there must be one value calculated for "low", if bins is two or more, then the first point is located at "low", and the last point is located at "high".  The step between the bins is (high-low)/(bins-1).  Examples:
```
    bins = 2, low = 1.0, high = 6.0
       values calculated at 1.0 and 6.0
```
```
    bins = 3, low = 1.0, high = 6.0
       values calculated at 1.0, 3.5, and 6.0
```
While the update can (in principle) be done directly on the GPU, and have the table directly accessed by the Cache::Manager, the results must be copied back to the CPU so that the Propagator event weight calculation can be done.  The current implementation does not expose the methods to hand the table directly from the external library to the Cache::Manager kernels.  The calculation can also be done solely on the CPU, and the results will be copied to the GPU.

# Library function to index events

The binning function is called as the MC events are read into the internal structures, and return an index into the lookup table for an event.  The `binningFunction` signature is:
```
    extern "C"
    double binFunc(const char* name,
                   int nvar, const double varv[],
                   int bins);
```
* name -- table name
* nvar -- number of (truth) variables used to find bin
* varv -- array of (truth) variables used to find bin
* bins -- The number of bins in the table.

The function should return a double giving the fractional bin number greater or equal to zero and LESS THAN the maximum number of bins.  The  integer part determines the index of the value below the value to be interpolated, and the fractional part determines the interpolation between  the indexed bin, and the next.  This determines where to find the entry for  the input (truth) variables.

# Compiling the library

 The library should loadable by dlopen, so it needs be compiled with (at least)

`gcc -fPIC -rdynamic --shared -o <LibraryName>.so <source>`
