# Tabulated Dials

A kriged dial provides an efficient way to do an event-by-event reweight of
an event based on a truth variable and fitting parameters when the lookup
is indexed by more than one truth variable. For instance, this can be used
for atmospheric neutrino oscillation probability tables where the event
probability depends on both the neutrino direction, and the neutrino
energy.  The table will be refilled fore each set of oscillation
parameters.  An external library is used declare a look up table, to fill
the table for each iteration, and look up the position of each event in the
table based on truth variables.

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
          weightFunction: <bin-func-name>  Function to find bin index
          weightVariables: [<var1>, <var2>, ... ] Variables used to fill
                                               the arrays of table indices
                                               and constants used to calculate
                                               the event weight
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

The function should return less than or equal to zero for failure, and otherwise, the number of elements needed to store the table (usually, the  same as the input value of "bins").

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

While the update can (in principle) be done directly on the GPU, and have the table directly accessed by the Cache::Manager, the results must be copied back to the CPU so that the Propagator event weight calculation can be done.  The current implementation does not expose the methods to hand the table directly from the external library to the Cache::Manager kernels.  The calculation can also be done solely on the CPU, and the results will be copied to the GPU.

# Library function to look up the events

The weight function is called as the MC events are read into the internal structures, and returns the indices and weights to use for an event.  The `weightFunction` signature is:
```
    extern "C"
    int weightFunc(const char* name, int bins,
                   int nvar, const double varv[],
                   int maxEntries,
                   int index[], double weights[]);
```
* name -- table name
* bins -- number of entries in the look up table
* nvar -- number of (truth) variables used to find bin
* varv -- array of (truth) variables used to find bin
* maxEntries -- The space allocated for index and weights
* index[] -- (output) an array filled with the index values used to weight
                 the event.  It must hold at least `maxEntries` values
* weights[] -- (output) an array filled with the weights to be applied the
                 the table entries referenced in the index array.  It must
                 hold at least `maxeEntries` values.

The function should return an int giving the number of entries that were filled in the index array.  The index and weights arrays are "parallel", so the first element of the index array is the index in the table that the weight in the first element of the weight array will be applied to.  The pseudocode for the event weight calculation is
```
double weight = 0;
for (int i=0; i<entries; ++i) weight += weights[i]*table[index[i]];
```

# Compiling the library

 The library should loadable by dlopen, so it needs be compiled with (at least)

`gcc -fPIC -rdynamic --shared -o <LibraryName>.so <source>`
