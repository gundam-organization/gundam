# Real-input runtime profile

`BackendRuntimeProfile.py` loads one `FitterEngine` once and retains its events,
physics dials, data and selected backend for all measurements. It does not run
Migrad, Hesse or a parameter scan. Configuration initialization is retained;
plot generation and post-fit output are disabled. No separate Original-engine
process is launched. GUNDAM is imported from the normal Python environment.

Rebuild and reinstall GUNDAM **including the Python interface** before running:
this change adds CUDA gradient stage counters and their Python bindings. Missing
bindings are detected before loading the dataset. Use a Release build on the CC.
The C++/Python integration can be tested on CPU, but CUDA compilation and GPU
timer measurements require a CUDA build and GPU allocation.

From the input directory, with `repo` pointing at the updated repository on the CC:

```bash
python3 -u "$repo/tests/python/benchmarks/BackendRuntimeProfile.py" \
  -c configOA2024.yaml \
  -of override/v13Prod.yaml \
      override/tests/ebMigrationDial.yaml \
      override/tests/binResponse.yaml \
      override/tests/gradientTest.yaml \
      override/tests/backendCudaTest.yaml \
  -t 4 -a \
  --output "$PWD/benchmark-cuda-runtime-$(date +%Y%m%d-%H%M%S)"
```

The output directory must be new. `--work-dir /path/to/GundamInputOA2024` allows
launching elsewhere; relative input paths are resolved there. The backend defaults
to CUDA. Its configured options (including propagation mode and weight layout)
are preserved. `--backend CPU` is available for small local smoke tests.
Redirect stdout/stderr to a file if a complete native initialization log is wanted.

## Workloads

All three scenarios run with value-only, gradient-only, and value-then-gradient
callbacks by default (`--scenarios` and `--workloads` select subsets):

- `constant`: repeated central point with resident caches already primed.
- `local1`: one coordinate displaced from the center. A value callback resets to
  the center **outside timing before every sample**, so only the selected
  coordinate changes in the measured propagation. The gradient still contains
  every fit coordinate. This is a cost probe, not a finite-difference test.
- `global`: alternate perturbations of all selected coordinates, with no reset
  between samples. This stresses broad response updates and migration.

`valueThenGradient` calls value followed by gradient at the same point; its
second propagation normally benefits from prediction reuse. `gradient` calls
have their own propagation and do not receive a preceding value at that point.
Local resets are the exception described above, at the central point.

Defaults: 3 trials, 30 calls per constant/global trial, and one complete sweep of
eligible fit coordinates per local trial. For 843 coordinates this means about
8,100 measured local workloads plus untimed resets. `--local-calls 100` samples
the first 100 eligible coordinates instead. `--parameter-match 'regex'` selects
coordinates to perturb; `--amplitude .01` is in parameter standard deviations.
One-sided in-domain proposals are allowed near boundaries; coordinates with no
representable in-domain proposal are excluded. A global sequence can leave a
one-sided coordinate unchanged between consecutive points. These deterministic
probes are not a replay of Migrad's actual call distribution.

The resolved configuration, parameter names, proposal coordinates, active module,
source revision, initialization time, GPU identity and CUDA diagnostics are saved.
The first value and gradient calls are recorded separately, including lazy gradient
model packing/allocation/upload. Then each workload is warmed up before timing.
No event/histogram materialization is requested in measurement loops.

## Outputs and timer interpretation

- `report.md`: readable medians/p95 and per-workload GPU stage rankings.
- `summary.json`: initialization, first-use costs, aggregate callback metrics,
  counts, environment, completion status and any Python exception.
- `samples.jsonl`: flushed after every sample; separate value/gradient snapshots,
  process CPU time, callback wall time, counters and checksums.
- `local-coordinate-costs.json`: coordinates sorted by median callback cost,
  with stage timings, migration counts and response candidate counts.
- `parameters.json`, `resolved-config.json`: exact workload description.

Wall time brackets the native callback, including its Python argument/result
conversion. Combined workload wall time is the sum of its two callback intervals;
Python timing-summary inspection and report writing are excluded. Process CPU
time counts all process threads; it does not measure GPU execution.

Propagation CUDA stages are the existing stream event intervals: parameter upload,
dial responses, event weights, fused observables/hard-bin search, fractional
contribution construction/indexing, histogram accumulation, and device likelihood.
Binning includes count/scan/sort and allocation synchronization. Counters show
recomputed events, response candidates, migration candidates/contributions,
allocated contribution bytes and readback volumes.

New gradient intervals surround producer Jacobian upload, dial derivatives,
likelihood bin adjoints, event derivatives (including migration and fractional-bin
derivatives), parameter reduction, and gradient readback. They reuse completed
propagation events and the existing gradient completion fence; no synchronization
is inserted between kernels. These extra event records introduce a small profiling
cost. Timings are available through `BackendTimingSummary`; aggregate
`BackendTimingTotals` still describes propagation, not gradients.

`gradientWallSeconds` excludes preceding propagation and backend-manager work.
`gradientModelPreparationSeconds` is normally zero after first use.
`gradientDeviceSeconds` includes upload through readback, including stream gaps;
these are **not** utilization, bandwidth or occupancy measurements. Empty stages
can show small event/launch overheads. CPU mode does not provide gradient stage
timers or GPU stage rankings.

Do not sum nested or overlapping timers: GPU intervals overlap native wall time,
manager preparation and cache preparation are host measurements, and event
selection is already included in cache preparation. Look first at total callback
wall time, then compare propagation and gradient totals, then their stage rankings.
A large remaining host cost warrants a CPU sampling profile; a dominant GPU
stage warrants an Nsight profile to distinguish bandwidth, occupancy and divergent
execution. This script provides stage attribution without requiring Nsight.

Results are checked for finite likelihoods/gradients and matching gradient size.
Checksums are diagnostic; this benchmark does not establish analytic-gradient
accuracy or compare backends numerically. Those checks remain in the dedicated
gradient tests and `GradientPerformance.py`.
