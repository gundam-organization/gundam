# Real-input runtime profile

## CUDA PythonWorker / nu-waves on FD OA 2026

Rebuild GUNDAM and its Python interface first. Inside a GPU allocation, activate
the environment normally used for these inputs, including `NU_WAVES_LIB` if
needed. Select the matching GUNDAM Python module (for a build-tree module,
export `PYTHONPATH="$build/src/PythonInterface:$PYTHONPATH"`). The configured
worker `pythonExecutable` must also have nu-waves and a Torch wheel supporting
the GPU; on a V100, check that it supports compute capability 7.0.

From the **CC copy** of `gundam-fd-oa-2026`, with `repo` pointing at this repository:

```bash
python -u "$repo/tests/python/benchmarks/BackendRuntimeProfile.py" \
  -c config/fdOa2026.yaml \
  -of override/enablePMNS.yaml override/withBeamFlux.yaml \
      override/tests/enableAnalyticGradient.yaml override/tests/backendCuda.yaml \
  -a -t 3 --toy -s 1234 \
  --profile-python-workers --parameter-match PMNS \
  --scenarios constant local1 global --calls 10 --trials 2 \
  --output "$PWD/benchmark-cuda-pmns-$(date +%Y%m%d-%H%M%S)"
```

This reproduces the configuration, thread count, Asimov source, toy mode and
seed of the corresponding `gundamFitter` command. `--toy [index]` has the same
optional throw index, defaulting to -1; `-a` remains effective as the data-source
selection when combined with `--toy`. This is a callback benchmark: it does not
reproduce Migrad's trajectory or total fit time. Plotting and Hesse are disabled.
Initialization follows the input configuration and is timed separately.

`--parameter-match PMNS` perturbs only coordinates whose full title contains
`PMNS`; it does not freeze other parameters, disable their dials, or truncate
the gradient. Inspect `parameters.json` to confirm the selected coordinates.
Remove this option for a second measurement that changes all fit coordinates.
Use separate output directories for each measurement.

`--profile-python-workers` enables `GUNDAM_PYTHON_WORKER_PROFILE` for the worker
subprocesses. Their `python-workers/worker-<pid>.jsonl` files record the actual
device, dtype, number of buffer rows, parameter count, script, command and
whether gradients were requested. `summary.json` retains initialization records.
Missing records cause an error rather than silently reporting zero worker time.
The environment variable can also be set to an existing absolute directory when
profiling `gundamFitter` directly, including a real fit.

Each callback adds these metrics to `samples.jsonl` and `report.md`:

- `workerCalls` / `workerWithGradientCalls`: actual evaluations, including cache
  behavior. Zero means no worker evaluation occurred during that callback.
- `workerWallSeconds`: sum of user `run(command)` times **including device
  completion**, using the synchronization already required by the worker protocol.
- `workerRunSeconds` / `workerSynchronizeSeconds`: Python call and subsequent
  synchronization. This split is not CPU versus GPU: Torch can synchronize inside
  `run`, and GPU work overlaps Python execution.
- `workerWallFraction`: worker time divided by callback wall time, per sample.

A large fraction in changing-PMNS workloads points to the worker as the dominant
callback cost. A small fraction directs attention to the existing backend timers
and host preparation. Constant workloads mainly probe cache reuse. Compare value
and gradient workloads to expose the additional oscillation Jacobian cost; a
value-then-gradient workload also captures reuse between the two callbacks.

The worker timer includes oscillator construction and any transfers inside user
code; it does not isolate `oscillator.probability` kernels. It excludes IPC,
buffer attachment, GUNDAM's buffer preparation and host readback after the reply.
In particular, onDevice gradients can still be read back by the host dial API.
Records are written to the node's temporary directory and copied to
`python-workers/` at exit, avoiding per-call writes to shared output storage.
JSON writing adds profiling overhead inside callback wall time; reading the
records and generating reports happen outside it. Repeat without profiling
to assess that overhead. These measurements establish timing attribution, not
numerical correctness or GPU utilization.

The FD configuration's `useBinnedWeights: true` evaluates nu-waves on the PMNS
grid rows, not on each MC event. The logged row count is therefore the relevant
oscillation batch size. Check `device=cuda` in the worker records and the nu-waves
startup log before attributing timings to GPU oscillation calculations.

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
time counts all parent process threads; it does not include worker process CPU
time or measure GPU execution.

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

## Checking the first cache optimization pass

After rebuilding/reinstalling the library and Python interface, a shorter probe
can reuse the full configuration with `--calls 10 --local-calls 64 --trials 1`.
This still loads the dataset exactly once, measures all workload types, and
visits the first 64 eligible coordinates. Compare these coordinates with the
same prefix of the baseline, rather than comparing against its complete sweep.
This is a quick performance check; a full production result remains necessary
before claiming a speedup for the fit as a whole.

Repeated identical host observable snapshots no longer force migration. In a
primed `constant` CUDA sample, `rebinnedEventCount` and
`eventWeightsRecomputedCount` should be zero. The CUDA statistical gradient can
also be reused: `gradientCacheHit = 1` and its GPU stage durations are then zero.
It is invalidated by prediction changes, and checked against raw parameters and
current producer Jacobians. Data histogram updates discard gradient buffers;
missing/invalid producer Jacobians still reject the request. Penalty gradients
and transformation to fit coordinates remain evaluated by the engine.

Loaded observables stay resident on the backend. Normal evaluations should report
`hostVariableReadCount = 0` and `hostVariableReadSeconds = 0`, including explicitly
dynamic samples. Host edits require `Sample.updateEventBinning()` or
`Propagator.updateEventSampleBinning()`; the next propagation reads and publishes
the edited values once. `managerSnapshotSeconds` measures the
whole snapshot construction, while `hostVariableReadSeconds` and
`hostVariableReadCount` isolate that read. `managerPreparationSeconds` retains
its earlier meaning (propagator preparation only). The host-read timer is nested
inside snapshot time, which is outside `propagationWallSeconds`.
