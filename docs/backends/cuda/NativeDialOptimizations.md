# Native Composite and Formula execution

CPU and CUDA build a common native program model from the propagation layout.
The source layout remains available; Metal continues to use its existing model.

The native model shares Composite instruction templates between events and stores
event-specific payload and Formula binding offsets separately. Formula preparation
folds constant arithmetic, including arithmetic on event-bound values, and shares
the remaining instruction sequence. Transcendental functions retain runtime
evaluation. Arbitrary supported expressions retain the interpreter fallback.

Common short formulas have direct execution plans. Eligible three-node chains
(Formula, scalar interpolation, Formula) execute with scalar intermediates rather
than graph scratch buffers. Mirrors and response supervision remain applied.
Multiply/add plans preserve separate rounding operations rather than introducing
fused multiply-add.

CUDA selects separate kernels for ordinary dials, generic programs, and models
whose program roots all support direct execution. This keeps the interpreter out
of ordinary and direct-program kernels. Program roots are grouped by parameter
dependencies and template; contiguous dirty ranges can be dispatched directly,
including ranges above the former half-model threshold. Event interleaving and
the rebinning implementation are unchanged.

CPU groups cached dials by their parameter dependencies, determines the dirty
count before selecting sparse or dense evaluation, and only constructs a dirty
index list when sparse evaluation needs it.

## Validation

The targeted suite contains 85 passing tests in both Debug and Release, including
bitwise comparisons against the original interpreter for eligible chains with
graphs, uniform/general splines, mirrors, and supervision. Synthetic Python tests
exercise CPU programs, Metal compatibility, and the benchmark reference/cleanup
options. GPU execution tests are included but require a CUDA build and device.
CUDA compilation and device execution were not available on the development Mac.

A local CPU benchmark with 277,534 events, three trials and 60 measured calls per
trial reduced the median global dial time from approximately 1.13 ms to 0.52 ms.
Total evaluation time changed from 5.31 ms to 4.69 ms. Endpoint event and histogram
digests and measured likelihoods were identical. These figures describe this CPU
workload; CUDA performance must be measured separately.
