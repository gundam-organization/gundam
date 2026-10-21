# MPS backend structure

The MPS backend executes numerical propagation on an Apple GPU through Metal.
It prepares a packed device model from the common backend representation and
retains GPU buffers for repeated evaluations. Parameters and other changing
inputs are supplied at propagation time; model structure is prepared during
the build.

Metal kernels use single precision. Converting published results to host doubles
does not recover precision lost during device calculations. This differs from
the CPU backend's double-precision execution.

The backend interface separates execution from host materialization, so callers
can request the outputs they need. The shared model and manager responsibilities
are described in [Backends library](../BackendsLibrary.md).
