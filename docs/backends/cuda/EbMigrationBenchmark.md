# Composite/formula migration benchmark

The script takes external benchmark inputs explicitly. No analysis configuration,
private dataset, override filename or experiment-specific variable is bundled or
selected automatically. From the repository, inside a CUDA GPU allocation, with
the matching GUNDAM Python module and ROOT runtime selected:

```bash
python3 tests/benchmarks/232CudaEbMigrationBenchmark.py \
  --work-dir /path/to/inputs \
  --config analysis.yaml \
  --override override/production.yaml override/selection.yaml override/migration.yaml \
  --output ./cuda-migration-benchmark
```

These paths are examples; provide your own inputs. `--config` is required.
Omit `--output` to create `./benchmark-232CudaEbMigrationBenchmark-YYYYMMDDHHmmss`
automatically, using local time at launch. An explicit `--output` path is used as supplied.
`--override` defaults to an empty list; files are applied in the given order.
Binning-response settings are read from these inputs, including any binning-response
override. No separate variable or smoothing-width option is normally needed.

No backend override is needed; provide binning-response overrides as ordinary input overrides. Each worker selects its
backend directly. Defaults compare CPU with cached/recomputed CUDA responses and
both event-major/interleaved layouts, using four threads. `deviceIndex=0` means
the first device visible through `CUDA_VISIBLE_DEVICES`.

By default, the only variant is **configured**: binning response, widths,
relative widths and dynamic-binning settings are kept exactly as supplied.

The loader also accepts legacy `fitterEngineConfig/propagatorConfig` and
`fitSampleSetConfig/fitSampleList` paths, including a modern likelihood block
containing only backend settings. Inspection does not rename or move keys: GUNDAM
handles compatibility when configuring the engine. Only the backend selection is
replaced; PCA, parameter-step scaling and plot configuration are not rewritten.

The observable recorded in snapshots is inferred from the enabled dials' single
`applyToVariable` target. `--variable` is only needed if several targets make this
ambiguous. No smoothing width or experiment-specific parameter filter is imposed.

Optional comparisons can be requested with `--variants hard smooth`:

- **hard:** disable binning response.
- **smooth:** use the configured binning response. An explicit `--smoothing-width`
  can optionally replace it for the inferred/selected observable.

Only these explicit comparison variants reset the explicit dynamic-binning request,
letting variable dials enable rebinning automatically and avoiding external-edit
polling. The default configured variant preserves that setting as well. Effective
settings and loaded dynamic/smoothed state are recorded in `coverage.json`.

All enabled samples are selected by default. Use `--sample-match REGEX` if some
samples do not bin the migrated variable. `--parameter-match REGEX` optionally
restricts effective fit coordinates; all eligible coordinates are selected by default.

## Workloads and timing

Default workloads are constant inputs, one changed coordinate, two changed
coordinates, and all selected coordinates. Local workloads cover every eligible
coordinate. They use reproducible bounded Gaussian displacements at 0.1 times the
parameter sigma (or step-size fallback). Defaults are three trials, eight warmup
calls, 100 constant/local calls and 40 global calls. No minimization or Hesse runs.

Each backend/variant gets a separate process and Asimov engine, retaining its
loaded data and GPU buffers throughout its trials. This is intentionally isolated:
the general profiler's `resident` backend replacement does not support these models
with observable migration. The default matrix loads five engines sequentially.

A shorter run:

```bash
python3 tests/benchmarks/232CudaEbMigrationBenchmark.py \
  --work-dir /path/to/inputs --config analysis.yaml \
  --cases CPU CUDA-cachedResponses-interleaved \
  --calls 20 --global-calls 10 --trials 1 \
  --output ./cuda-migration-smoke
```

Use `--variants hard smooth` only when requesting an additional controlled comparison. CPU must
remain in `--cases` as the numerical reference. `--cases CPU` is available for a
local script smoke test; its summary explicitly records that no GPU comparison ran.

## Outputs and validation

- Root `report.md`: response, observable/search, rebin/index and histogram times,
  total evaluation/GPU times, CPU speedups, migration coverage and smoothing overhead.
- `hard/report.md` and `smooth/report.md`: full stage/counter tables and layout sizes.
- `*/comparison.json`: CPU/CUDA numerical comparisons within each binning variant.
- `*/CASE/coverage.json`: actual samples/events, configured Composite/Formula counts,
  CUDA program metadata, and a host probe demonstrating fractional bin responses.
- `*/CASE/local1-coordinate-costs.json`: costs, cache candidates and processed events
  per coordinate, including coordinates that affect no events.
- `*/CASE/migration.json`: events with changed observable and hard bin index at each
  trial endpoint, relative to its center. A processed event need not change hard bin.
- Per-trial JSON and binary event snapshots; resolved configurations, Python module
  path/hash, environment/device metadata and `native.log` for each worker.

Numerical checks cover every measured likelihood and every trial endpoint's
histogram contents/errors, smoothed outside-bin contents/errors, all event weights,
the selected observable values and exact hard bin indices. Snapshots/readback/probes occur outside
measurement; a snapshot uses an additional untimed evaluation at the same point
with automatic materialization. Warmup snapshots are also retained. Large samples
can produce substantial snapshot files.

Use `--reference-output /path/to/previous-run` to also compare each new case with
the previous run's CPU reference. The script writes `comparison-reference.json`
for each variant. The previous run must retain its binary snapshots.
Use `--discard-event-snapshots` to delete the new run's `.bin` snapshots after
all numerical and migration checks succeed. Failed runs retain their snapshots,
and reference files are never removed. JSON reports and timings remain available.

When requested, hard and smooth models have separate Asimov data and are compared to their own CPU
references. Their likelihoods are not equated; their parameter manifests, replay
coordinates and loaded selected event layouts must agree. Defaults are relative
`1e-9` and absolute `1e-7` numerical tolerances, with exact bin-index agreement.

The script fails if it cannot exercise Composite/Formula programs on CUDA,
observable propagation, rebinning, or (when smoothing is configured) fractional responses
in the host probe. `--probe-events` controls the maximum probe size per sample.
Actual event movement is checked at measured endpoints. GPU fractional contributions
are validated through histogram agreement with CPU, not through the host probe.

Response timers combine Composite traversal, Formula nodes and interpolation;
these operations share a kernel and are not timed individually. CUDA variable/
search and binning/index stages have different boundaries from CPU stages. CUDA
stream times overlap host evaluation: compare total evaluations first and do not
sum the host/GPU columns. No additional per-opcode timing fences are introduced.

The script reuses `229CudaBackendProfiling.py` for replay, timings and numerical
comparison. It requires no new Python library. The automated test creates its own
six-event ROOT fixture and configuration in a temporary directory, then exercises
both CPU binning variants. It does not access
external benchmark inputs. CUDA execution must be checked on a CUDA machine.
