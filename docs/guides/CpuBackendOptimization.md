# CPU backend optimization — October 20, 2026

This work prepares the buffers and kernels for future specialization. It adds no JIT compiler or LLVM/ROOT dependency and does not reduce CPU floating-point precision.

## Implemented changes

- **Shared CPU model ownership.** `BackendManager` owns the layout and passes shared ownership through `Backend::buildShared()`. The CPU retains the same storage instead of copying its vectors. The model stays alive when the initial owner releases it. Standalone `build(const EngineView&)` retains its snapshot behavior. Other backends may retain their hardware-specific representations; Metal currently keeps its own snapshot and packing.
- **Contiguous reweight descriptors.** `PropagationView::eventWeights` stores weight descriptors indexed by `EventView::resultIndex`. It replaces the weight field previously embedded in `EventView`, without adding a second persistent copy. On this platform, the hot loop reads 24-byte descriptors instead of 88-byte event structures. Binning and observable metadata are separate. CPU, Metal and reference consumers were updated.
- **Isolated weight kernel.** `CpuWeightKernels::reweight()` accepts buffers, a range and a response evaluator. Four precompiled variants cover cached or mixed responses, with or without a reweight cap. Construction selects one variant, called once per worker task. The mixed path retains support for external weights and other non-cacheable dials.
- **Constant responses.** `Shift` dials are evaluated at construction, including their bounds and smoothing, and retained in the existing cache. Their position in the product is unchanged. Constants are not folded into the base weight, preserving floating-point operation order and cap placement.
- **Prepared CPU ranges.** Each thread receives a contiguous result range, balanced using an estimated cost of `1 + dialCount`. No per-thread copy of input arrays is added. The estimate does not yet distinguish the costs of individual dial types.
- **Less repeated work.** The cached-dial count is computed at construction instead of scanning every dial on every propagation for statistics. Output weights retain their size and are overwritten without an intermediate zeroing pass. The need for event binning is determined at construction. Static histogram filling bypasses per-event checks intended for dynamic binning; dynamic and smoothed paths remain available.
- **Accumulated measurements.** `BackendTimingTotals` accumulates propagation counts and actual stage durations since `build()`. The fitter prints totals at the end of a fit. Per-call CPU timers now report the latest call rather than a moving average.

CPU parameter values, payloads, weights and results remain `double`. No offsets were narrowed in this change; event dial indices were already `uint32_t`.

## Real-data benchmarks

The local Release binary ran sequential Asimov fits with four threads and `--skip-hesse`, using the supplied Upgrade CC 2026 and OA2024 configurations and overrides. Separate post-fit Hesse was disabled; internal minimizer operations were unchanged. OA2024 runs Simplex followed by Migrad.

| Measurement | Upgrade before | Upgrade after | OA2024 before | OA2024 after |
|---|---:|---:|---:|---:|
| Minimization time | 79.035 s | 55.221 s | 17.889 s | 11.634 s |
| Minimization calls | 52,070 | 52,070 | 6,761 | 6,761 |
| Minimization time per call | 1.518 ms | 1.061 ms | 2.646 ms | 1.721 ms |
| Active allocations after initialization | 2.41 GB | 1.87 GB | 4.23 GB | 3.13 GB |
| Process peak RSS | 5.131 GB | 4.678 GB | 9.025 GB | 8.260 GB |

Observed minimization time reductions were **30.1% and 35.0%**. Active allocations decreased by approximately **22% and 26%**. Per-call times include minimizer overhead, rather than isolating the kernel. Allocation figures are rounded in the logs. Loading and output serialization may still dominate peak RSS.

Each case has one baseline execution, and other applications were running. These are observed improvements, not isolated measurements of each individual change or guarantees for other analyses.

Both models select the fully cached path:

| Layout | Upgrade | OA2024 |
|---|---:|---:|
| Events | 277,177 | 416,034 |
| Unique dials | 1,776,296 | 3,529,068 |
| Weight descriptor bytes | 6,652,248 | 9,984,816 |
| Event dial index bytes | 17,211,408 | 26,246,868 |

## Remaining costs

Totals include all propagations since backend construction, including initialization, minimization and output generation. They exclude preparation and materialization in the manager, so their counts differ from minimization call counts.

| Accumulated time after optimization | Upgrade, 52,702 propagations | OA2024, 8,452 propagations |
|---|---:|---:|
| Entire backend | 54.887 s | 13.1256 s |
| Dial response cache | 10.2553 s | 1.14999 s |
| Per-event response products | 37.9125 s | 9.46973 s |
| Histograms | 6.36111 s | 1.83847 s |
| Statistical likelihood | 0.348601 s | 0.664735 s |

Response products still account for approximately 69% and 72% of backend time. Removing branches does not remove indirect response loads, which remain a priority. These measurements alone do not demonstrate memory-bandwidth saturation.

## Validation and limits

- Release compilation succeeded.
- **105 C++ tests passed**, excluding the `RootReaderCleanup` family. That family exits on a configuration missing a required `name` field; the same failure was reproduced using the Debug binary built before these changes. The complete suite is therefore not fully green.
- **26 Python tests passed**: all existing Python scripts numbered 210 through 229 in `tests/fast-tests`, including `215ArrayDialBranchData.py`, CPU/Metal propagation, external weights, observable responses and dependencies, dynamic binning, response limits and smoothing.
- New CPU tests cover fully cached and mixed paths, external and constant dials, cap placement with a negative base weight, uneven workloads, 1/2/5 threads, reordered metadata, changed parameters, retained output storage, empty models, NaN, rebuilds and shared ownership.
- Before and after optimization, both backend fits converged with identical call counts, printed final likelihoods and saved parameter states.
- Exact comparison found no differences in the contents and variances of **1,419 Upgrade histograms and 10,375 OA2024 histograms**, including ROOT cycles. Names containing memory addresses were normalized, and resulting name collisions were compared as multisets of histogram contents.
- The OA2024 diagnostic about a bin empty in both data and MC was present before and after optimization.

These comparisons validate the optimized backend against its previous version with separate Hesse disabled. They do **not** establish equivalence between the backend and reference propagation paths for post-fit Hesse uncertainties. A subsequently reported discrepancy in the 320 kA flux uncertainties is under investigation.

Mixed and dynamic features were checked for correctness, but their performance gains have not been quantified on dedicated large fits.

## Follow-up: post-fit Hesse stability

The supplied `comp.root` and original fit outputs showed matching Migrad uncertainties but Hesse flux error ratios as large as 1.4328734 between backend and reference CPU propagation. Both Hesse runs reported status 0 and covariance status 3.

Debugger inspection identified cancellation in the shared `BarlowBeestonBanff2022` likelihood, selected by `BarlowLLH_BANFF_OA2021`. For one nominal bin, data and prediction were both `655.82092932695639`, while the Barlow-Beeston adjusted prediction differed by one rounding step. The expression `newmc - data + data * (log(data) - log(newmc))` produced a negative statistical contribution of `-1.1368683772161603e-13`.

The two CPU paths accumulate histograms in different orders. Real-model probes found bin differences around `3.6e-12` and statistical-likelihood differences around `1e-12`. Curvatures agreed at ordinary probe steps and became sensitive to rounding at smaller steps. Hesse differentiates and inverts this noisy likelihood, so agreement of fitted values or histograms at ordinary tolerances was insufficient to validate uncertainties.

The shared likelihood now evaluates the equivalent `data * (r - log1p(r))`, where `r = (newmc - data) / data`. Near zero, a series through `r^10` avoids cancellation in `r - log1p(r)` itself. Extreme ratios retain the separate-log formulation. The mathematical likelihood, priors, event weights and histogram accumulation algorithms are unchanged; the numerical evaluation is more accurate near the minimum. Uncertainties from the old reference path can therefore change too.

Both supplied Upgrade configurations were rerun in Release with four threads and Hesse enabled, using isolated outputs:

| Validation | Result |
|---|---:|
| Maximum relative Hesse error difference, flux parameters | 4.83e-9 |
| Maximum relative Hesse error difference, all parameter sets | 1.40e-8 |
| Saved best-fit parameter value difference | 0 |
| Relative Frobenius difference of the full covariance matrix | 1.22e-9 |
| Hesse calls, reference / backend | 50,539 / 50,539 |
| Hesse status, reference / backend | 0 / 0 |
| Covariance status, reference / backend | 3 / 3 |

The C++ near-minimum regression fails before the fix and passes after it. `230AsimovHesseStability.py` checks small-step curvature and post-fit Hesse uncertainties against an analytic one-parameter Asimov model, with and without the CPU backend. The resulting validation covers **106 C++ tests** (with the pre-existing `RootReaderCleanup` exclusion) and **27 Python tests** numbered 210 through 230.

Raw outputs, logs and comparison data are under `/tmp/gundam-hesse-investigation/`. The regenerated comparison is `comp-stable.root`; the supplied comparison and fit outputs were not overwritten. These Hesse checks apply to CPU propagation; the earlier performance table remains a historical measurement of the optimization before this numerical fix.

## Proposed next steps

1. Resolve the reported backend/reference Hesse discrepancy and add a regression test for likelihood curvature and post-fit uncertainties.
2. Build a synthetic benchmark matrix covering event counts, dials per event, dial sharing, changed parameters, external weights and dynamic binning. Five times as many events and five times as many dials per event can produce 25 times as many occurrences.
3. Measure response locality and indirect loads. Investigate index and response placement without duplicating data or changing floating-point product order.
4. Select kernels per block for mixed models whose individual blocks use only cached responses. Currently, a non-cacheable event dial selects the generic path for the whole model.
5. Review dial metadata and occurrence bindings. `EngineBindings::eventDials` is populated, but no internal reader has been identified; verify its purpose and cost before removing it.
6. Consider dynamic specialization using the isolated kernel and the same buffers. `TInterpreter` is not planned; LLVM ORC could be an optional CMake dependency.

Local logs, RSS samples, commands, ROOT comparisons and measurement scripts are archived in `cmake-build-release/performance-baselines/2026-10-20/`. Large ROOT outputs remain in `/tmp/gundam-memory-baseline-20261020/`. This optimization work did not replace the binary installed under `Work/Install`.
