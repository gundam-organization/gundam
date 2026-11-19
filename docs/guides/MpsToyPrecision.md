# MPS toy precision — October 24, 2026

## Failure and cause

The Upgrade CC 2026 toy with seed 1234 converged with CPU but failed with the
default MPS backend during Migrad. The existing float-epsilon function precision
recommendation had been validated on Asimov Hesse, not on this toy.

The resident GPU executor evaluated responses in float pairs, then rounded each
response to float32 before multiplying event weights. Contribution fractions,
partial histograms and final histogram readback also crossed float32 boundaries.
The objective agreed closely in value but its finite differences did not.

At the failed MPS point, physical flux parameter #100 gave the following central
second differences (step in prior-sigma units):

| Step | Reference | Original MPS | Corrected MPS |
|---|---:|---:|---:|
| 0.01 | 328.497618 | 328.649780 | 328.497618 |
| 0.001 | 328.497618 | 341.439710 | 328.497614 |
| 0.0001 | 328.497617 | 953.438939 | 328.497322 |

A native CLion debugger stop after Migrad independently confirmed
`_fitHasConverged_ = false`, status 3, covariance status 1, EDM 10327.2102 and
minimum 506.501494 in the Debug build. The Release reproduction below differs
slightly in its trajectory. No user breakpoints were changed.

## Correction

The resident event executor now preserves float pairs through base weights,
response products, reweight caps, smoothed fractions, squared weights, both
histogram reductions and readback. Host values combine both components in double
precision. Buffer allocations, transfers and device-view sizes follow the new
representation. Event-weight and histogram scalar buffers double in size;
contribution records grow from 12 to 16 bytes. Response and observable buffers
already used float pairs.

The resident executor recommends relative function precision `1e-12`. This is a
conservative numerical differentiation setting for the increased precision, not
a model-independent likelihood error bound. Explicit configuration still wins.
The fixed-bin packed executor retains float32 outputs and float epsilon.

## Real toy validation

Working directory: `upgrade-cc-2026`. Arguments:

```sh
-c ndUpCcInc.yaml \
-of override/datasetHighlandv5.27.1_CCLyon.yaml override/backendMpsTest.yaml \
-a -t 4 -s 1234 --toy --skip-hesse --light-mode
```

The CPU run substitutes `override/backendCpuTest.yaml`. Outputs and logs are
isolated under `/tmp/gundam-toy-mps`; the user's existing outputs were preserved.
Corrected runs use `cmake-build-release/src/Applications/gundamFitter` with
`DYLD_LIBRARY_PATH` unset to avoid installed libraries.

| Run | Migrad status | Covariance status | Minimum | EDM | Calls | Migrad time |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 0 | 3 | 361.423467 | 0.00141189 | 294751 | 283.16 s |
| Original MPS | 3 | 1 | 506.573354 | 10399.4670 | 54703 | 35.27 s |
| Corrected resident MPS, automatic `1e-12` | 0 | 3 | 362.077557 | 0.000970931 | 158497 | 54.32 s |
| Corrected resident MPS, explicit `1e-14` | 0 | 3 | 362.010500 | 0.00182325 | 168075 | 49.11 s |

These are single runs on a shared machine, not performance benchmarks. Hesse was
skipped in every real toy fit. Merely setting `1e-12` on the old float32 executor
also converged, at 362.081472, but left the finite-difference noise in place.
Setting `1e-9` still failed, with EDM 10.0345.

The distinct minima remain a limitation. At identical parameter points, corrected
MPS and the original reference engine agree:

| Injected point | Reference objective | MPS minus reference |
|---|---:|---:|
| CPU minimum | 361.423466721091 | 3.41e-13 |
| Corrected MPS automatic minimum | 362.077556955000 | -1.25e-12 |
| Corrected MPS `1e-14` minimum | 362.010499516017 | -2.27e-13 |

This supports a minimizer trajectory / differentiation-step explanation for the
remaining minimum difference; it does not establish that each reported minimum
is global or that all seeds converge.

## Regression checks

- Release build succeeds; 70 targeted C++ backend, device and binning tests pass.
- Python tests 212 (propagation), 214 (external weights), 221 (migration / smoothed
  binning / cache / readback), 229 (response smoothing), 231 (analytic Hesse) and
  236 (toy precision) pass.
- New `236MpsToyPrecision.py` checks nonzero Poisson likelihood gradients and
  curvatures at multiple steps against analytic values, then checks a toy
  normalization minimum against its analytic solution and CPU.
- The same new test fails with the original installed library: at step 0.001,
  curvature is 1094.939435 instead of 1092.561983.
- Test 214 had an existing stale transfer-size assertion (4 rather than 8 bytes
  per external value in resident execution). It failed identically before this
  correction; its assertion now describes the existing float-pair upload.
