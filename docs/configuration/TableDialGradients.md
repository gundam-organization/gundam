# Tabulated and Kriged dial gradients

`Tabulated` and `Kriged` can use an optional derivative function supplied by
their external table library. Add its exported symbol to the existing
`tableConfig`:

```yaml
tableConfig:
  # Existing libraryPath, initFunction, updateFunction, etc.
  gradientFunction: differentiateTable
```

The ABI is the same for both dial types:

```cpp
extern "C"
int differentiateTable(const char* name,
                       double derivatives[], int bins,
                       const double parameters[], int nParameters);
```

Return zero on success and a nonzero status on failure. Fill **every** entry:

```cpp
derivatives[iEntry * nParameters + iParameter] = /* dT[iEntry]/dParameter */;
```

Parameters are the current dial inputs, in `dialInputList` order. Derivatives
must be with respect to these inputs, before GUNDAM applies parameter mirrors
or transforms to minimizer coordinates. The function supplies analytic or
otherwise automatically differentiated derivatives of the table producer;
GUNDAM does not estimate them by finite differences.

For example, for `T[i] = (i+1)*p[0]*p[0] + (i-1)*p[1]`:

```cpp
extern "C"
int differentiateTable(const char*, double* derivatives, int bins,
                       const double* p, int nParameters) {
  if (nParameters != 2) return 1;
  for (int i = 0; i < bins; ++i) {
    derivatives[i * nParameters] = 2. * (i + 1.) * p[0];
    derivatives[i * nParameters + 1] = i - 1.;
  }
  return 0;
}
```

A gradient request updates the values and derivatives once per table before
parallel event evaluation. Value-only propagation calls `updateFunction` without
calling `gradientFunction`. Both functions must describe the same deterministic
table at the supplied parameters. Derivative storage is initialized to NaN;
incomplete or non-finite output, and nonzero producer status, raise an error.
Any new value update invalidates the previous derivatives. Reads also verify
that the stored input point and table dimensions match.

Event interpolation uses the same coefficients as response evaluation:

- `Tabulated`: `dR = f*dT[i] + (1-f)*dT[i+1]`, with the stored fraction `f`.
- `Kriged`: `dR = sum(a[j]*dT[index[j]])`.

Indices and interpolation coefficients are fixed event information, not fit
parameters. Observable-dependent interpolation/migration is not differentiated.
Response limits and response smoothing apply afterward through the supervisor's
chain rule.

Existing libraries without `gradientFunction` remain usable for value-only
fits. Requesting analytic gradients for those dials fails during validation.
An explicitly configured symbol that cannot be loaded fails during factory
construction.
