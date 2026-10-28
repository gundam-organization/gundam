# CPU formula dials

Standalone `Formula` dials and formula nodes in composites are compiled into
backend bytecode when the engine layout is built. The CPU interpreter uses immutable programs and preallocated
scratch per worker; propagation does not call `TFormula`. Existing dial
construction still uses ROOT to validate formulas and bind their named inputs.
Composite formulas can mix parameters, upstream dial responses and loaded event
variables, including inside nested composites. Event values retain their loaded
snapshot semantics; they are not read back from an observable modified by propagation.

The configuration syntax is unchanged:

```yaml
dialType: Formula
dialInputList:
  - {name: delta, parameterName: delta}
  - {name: alpha, parameterName: alpha}
  - {name: q3, eventVariable: q3}
options:
  formulaStr: "[delta] + [alpha] * (0.056 * [q3] - 41.6)"
```

The supported expression subset includes named inputs `[name]`, numeric literals
(including scientific notation), parentheses, unary `+` and `-`, arithmetic
`+ - * /`, and right-associative powers `^` and `**`. Power has higher precedence
than unary minus: `-2^2` evaluates to `-4`.

Supported functions are `exp`, `log`, `log10`, `sqrt`, `abs`, `fabs`, `sin`, `cos`,
`tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `erf`, `erfc`,
`floor`, `ceil`, `pow`, `min`, and `max`. Common `TMath::` and `std::` spellings
are accepted, including `TMath::Power` and `TMath::Pi()`. The compiler also
recognizes `pi`, `twopi`, and `e` as numeric constants, subject to the existing
ROOT frontend accepting the expression.

Parameter inputs preserve their configured ordering and mirroring. Loaded event
inputs are bound separately for each dial and share the underlying formula
program. Binning constants, response limits, smoothing, and observable targets
retain their existing behavior. Formulas without parameter inputs are evaluated
once when the CPU backend is built.

The CPU response cache indexes multi-input dials by their parameter dependencies.
Only lists belonging to changed parameters are visited; generation markers
deduplicate dials reached through several changed parameters or repeated inputs.
During formula evaluation, named input bindings are resolved once into a
contiguous worker-local array before interpreting the bytecode. The worker's
preallocated formula scratch contains both this array and the evaluation stack.

Unsupported syntax (including conditionals, comparisons, positional `x[i]`, ROOT
distribution shortcuts and arbitrary C++ calls) raises an error during layout
construction. There is no fallback to ROOT evaluation. Floating-point results
are compared with ROOT within numerical tolerances, rather than guaranteed to
be bit-identical.

The compiler limits expressions to 4096 instructions and 128 nested unary/parser
levels. Bytecode validation checks opcodes, operand bounds, stack underflow,
required stack size and a single final result before execution. GPU execution
is not implemented; fixed-width instructions contain no host pointers and can
be packed for a future device interpreter.
