# Shared formula and composite execution

The formula compiler, bytecode and model validation remain common to all backends.
Execution now shares these additional pieces:

- `DeviceMath/SharedFormulaOperations.h` defines the unary/binary opcode dispatch
  inventory once. It is deliberately included without a guard under case-generation
  macros, and embedded into the Metal source by CMake.
- `Semantics/NativeFormulaProgram.h` implements the CPU/CUDA interpreter. CPU passes
  its existing `FormulaProgram`; CUDA passes a non-owning view over packed tables.
- `Semantics/NativeCompositeProgram.h` implements CPU/CUDA graph traversal, input
  mirrors, node supervision and graph/spline evaluation. Stateless policies select
  formula bindings, surface implementations and error handling at compile time.

CPU retains its contiguous input gathering before each formula, per-worker scratch,
original surface routines and response cache. The native interpreter keeps the
original CPU calling convention when outlined by the compiler. Small policy adapters
are forced inline, including the surface adapter. CUDA retains native-double device
math, per-root device scratch, packed tables and its spline conversion guards.
The guards are compiled out of the CPU specialization with `if constexpr`.

Metal keeps its device address spaces, value-based DoubleFloat stack operations,
compensated math and composite traversal over packed DeviceProgramInput descriptors.
It uses the shared opcode inventory with FormulaMath adapters. Sharing the graph
traversal with Metal would require further changes to these storage interfaces;
this refactor leaves that hardware-specific path intact.

No allocations, virtual calls, runtime backend selection or new dependencies are
introduced into the interpreters. The existing CPU/CUDA wrappers remain available.

## Validation (2026-10-28)

Release build on Apple M4 Max, four threads, production `ndUpCcInc.yaml` with the
Highland/ebMigration/binResponse/backend overrides and `onlyEb.yaml`. Each run uses
three trials of 60 measured calls, eight warmup calls, and separate loaded engines
for the original and refactored code. CPU uses `230CpuFormulaCompositeBenchmark.py`;
Metal uses `225MpsBackendProfiling.py` with 60 global calls and no profiling probes.

| Measurement, median milliseconds | Before | After |
|---|---:|---:|
| CPU response evaluation, C nu | 1.104 | 1.086 |
| CPU response evaluation, alpha | 1.105 | 1.089 |
| CPU response evaluation, all Eb | 1.147 | 1.131 |
| CPU full evaluation, all Eb | 5.319 | 5.300 |
| Metal device propagation, local1 | 0.329 | 0.332 |
| Metal device propagation, local2 | 0.815 | 0.812 |
| Metal device propagation, global | 0.822 | 0.820 |

These small differences do not establish a general speedup. They show no material
regression on this workload. An interleaved CPU microbenchmark and assembly inspection
also checked the interpreter boundaries and elimination of policy calls. Benchmarks
include the shared composite/formula path, with rebinning reported separately.

CPU histogram/event digests and every measured likelihood agree exactly with the
original version. Metal's maximum absolute differences over all measured trials are
`3.73e-13` for likelihood and `1.28e-11` for histogram values/errors.

83 targeted C++ tests pass, including all opcodes, CPU bitwise composite comparisons,
CUDA packing/native interpreter tests and Metal execution. The Python formula/composite
matrix also passes against ROOT on Metal in both propagation modes. Existing tests
cover mirrors, smoothing, event-bound inputs, nested graphs and observable migration.

CUDA device compilation, register use and timings still require a CUDA machine. Run
`ctest -R '^cuda-backend-'` and the existing CUDA benchmark wrapper documented in
`cuda/FormulaCompositeDials.md` at CCLyon before drawing CUDA performance conclusions.
