# Profiling the CPU backend on a batch worker

The CPU backend can export sampled diagnostics to a standalone JSON Lines file.
This is intended to investigate scaling on large models without requiring ROOT
to read the report, privileged counters or an extra profiling library.

## Running the probes

Build and install the instrumented revision in the cluster environment as usual.
Check the fitter's version at startup to ensure the job uses the new binary.
Copy [CpuProfiling.yaml](../../resources/examples/CpuProfiling.yaml) to an override
such as `override/cpuProfile4.yaml`:

```yaml
fitterEngineConfig:
  likelihoodInterfaceConfig:
    backendManagerConfig:
      isEnabled: true
      useBackend: CPU
      options:
        - name: CPU
          profileOutputPath: cpu-profile-4.jsonl
          profileSampleInterval: 100
          profileMaxSamples: 200
```

Run the full-data OA2024 configuration and its usual overrides, adding this
profiling override last. For example, with the local input subset:

```sh
gundamFitter -c configOA2024.yaml \
  -of override/v12ProdRun45.yaml override/backendCpuTest.yaml override/cpuProfile4.yaml \
  -a -t 4 --skip-hesse -o cpu-profile-fit-4.root > cpu-profile-fit-4.log 2>&1
```

For the grid, use the overrides that select **all** simulated data. The example
above only identifies the known local configuration; it does not select the
full-scale dataset. Keep the usual minimization enabled: a dry run does not
exercise the representative parameter changes. `--skip-hesse` avoids spending
time on the separate post-fit error calculation and can be omitted when that
phase is of interest.

Repeat with `-t 8`, `profileOutputPath: cpu-profile-8.jsonl` and separate log/fit
outputs. Prefer the same node and CPU allocation, allowing at least eight CPUs
for both executions. Run sequentially with identical input and minimizer
settings. If inexpensive, also measure 1 and 2 threads. Repeat runs help separate
job placement and shared-node load from actual thread scaling.

The PID is inserted automatically before the last extension: configuring
`cpu-profile-4.jsonl` writes `cpu-profile-4-<PID>.jsonl`. The actual path is
printed at startup and stored in the report metadata.

Return the `.jsonl` files and corresponding fitter logs. The fit ROOT files are
not required for this performance investigation. Stage these files back from the
worker's local scratch directory before the batch job exits.

## Configuration and output lifetime

| CPU `options` entry field | Default | Meaning |
|---|---|---|
| `profileOutputPath` | empty | Enables diagnostics when nonempty; relative to the process working directory; the PID is inserted before the extension. Parent directories must already exist. |
| `profileSampleInterval` | 100 | Sample every Nth backend propagation after the first five calls; strictly positive. |
| `profileMaxSamples` | 200 | Maximum samples per backend build, including startup; strictly positive. |

When disabled, no report is created and there are no per-dial diagnostic counters
or clock reads. There are small checks at job/stage boundaries. In sampled calls,
cache counters are enabled through a separate compiled loop; no clock is read
per dial. Startup scans collect model inventories only when profiling is enabled.

A metadata record is written at construction, followed by sampled propagation
records. Each record is flushed immediately, so completed records survive an
early job termination. Each process writes to its PID-suffixed filename. Existing files at that resolved
path are appended to, with a new `runId` per backend build. The unsuffixed
configured file is left untouched. A killed process can leave one partial
last line, which the summary tool ignores. Failure to open or write an explicitly
requested report raises an error.

The sample limit bounds record count, not an exact byte count. Reports will
usually be a few megabytes with these defaults; thread count and available OS
metadata affect their size. Metadata source files are capped at 65,536 bytes
each, so large CPU listings and NUMA maps may be incomplete. No full event or dial
payload is copied into the report.

## Reading a report

The summary tool uses only the Python standard library:

```sh
python3 /path/to/gundam/misc/summarizeCpuProfile.py cpu-profile-4-*.jsonl cpu-profile-8-*.jsonl
```

It excludes the first five backend calls from its summary by default; use
`--include-startup` to inspect them. All startup records remain in the raw file.
The first call can populate every cached response and is not representative of
a typical Minuit parameter step.

Backend call numbers include initialization and output production, and differ
from Minuit evaluation counts. Records include timestamps and time since profile
construction to relate them to fitter logs. Samples cover all subsequent phases
until the sample limit is reached; the report does not label fit phases itself.

## Recorded probes and interpretation

### Model layout and working set

The metadata records event counts, unique and referenced dial counts, per-type
inventories, cached dials per physical parameter, event dial-count distribution,
prepared fixed weight ranges and their estimated work. With dynamic weight
scheduling those ranges are only a reference; actual worker event/block counts
are recorded per sample. See [CPU weight scheduling](CpuWeightScheduling.md). It also records sizes,
capacities, base addresses and element sizes of the main model/cache buffers, plus expected
weight-output and per-thread histogram storage. These are selected buffers,
not total process memory or measured memory traffic.

Compare these working sets with the reported CPU cache sizes. Large indirect
response arrays can explain why a cluster CPU behaves differently from a local
machine, but footprint alone does not establish bandwidth saturation.

### Cache invalidation and serial preparation

Each sample records changed physical parameter indices, candidate dirty dials,
and the serial preparation duration before cache-worker dispatch. Workers report
how many responses were actually evaluated, how many transformed inputs were
unchanged, and the evaluated dial-type distribution.

This distinguishes many genuinely dirty responses from expensive preparation or
skipped evaluations. In particular, a change to one fitted eigenparameter can
change many physical parameters. Samples with different numbers of changed
parameters must not be treated as identical workloads. The summary groups cache
stage times by that count; the raw indices and per-type counts allow finer
comparisons.

### Per-thread work and synchronization

For the cache, event weights and histogram jobs, records contain each worker's
start and end offset relative to stage entry, elapsed execution time, and the
stage duration on the calling thread. Fixed weight ranges are prepared at construction; dynamic scheduling instead
claims from two automatically prepared blocks per worker and records the actual
counts for each worker. No numeric block-size configuration is needed. Cache ranges are equal contiguous partitions of the sampled dirty-dial list.
Histogram workers partition the event metadata by event count.

For the cache stage, start offsets include serial cache preparation. Subtract
`cachePreparationSeconds` when examining launch delay. Histogram stage job times
exclude histogram initialization and final reduction; compare them with the
full `backendTimingSeconds.histograms` value to assess these costs.

`joinTailSeconds` is the interval from the latest worker end timestamp to job
return. It includes accounting/dispatch overhead and is **not** a direct measure
of lock contention. A large spread in worker durations may indicate uneven dial
costs, scheduling differences or memory placement. It does not alone identify
which explanation applies.

On Linux, worker records additionally contain CPU time, voluntary/involuntary
context-switch and page-fault deltas, plus the logical CPU at job start and end.
CPU time much smaller than elapsed time suggests time off-CPU. Similar CPU and
elapsed times can still include stalls waiting for memory. Different endpoint
CPU IDs prove movement; identical IDs do not exclude an intervening migration.
Unsupported worker resource fields are `null`, including on the current macOS
implementation, rather than being substituted with misleading process counters.

### Host and batch-job constraints

Linux metadata includes CPU model information, allowed logical CPUs, core/socket
and SMT sibling IDs, selected cache sizes and sharing lists, available frequency
policy information, allowed NUMA nodes, node CPU lists/distances, initial NUMA mappings, and cgroup
membership/mount information. Samples add process RSS/status, pressure information and discovered cgroup CPU
accounting counters. More expensive memory-rollup and cgroup memory/NUMA reads
are collected on the first and sixth sampled calls and every 50th sample.
`systemSnapshot` distinguishes `light` and `full` snapshots, and
`exportReadSeconds` separates time reading process and cgroup files. CPU
accounting paths are discovered at construction and reused for light snapshots. NUMA maps are
sampled on the first and sixth sampled calls to capture buffer placement after
startup. Missing or inaccessible files are omitted.

Cgroup files are read from conventional mount roots and process-relative paths;
Ancestors are included up to 16 levels to capture job/step limits above the task.
Custom mounts/namespaces can prevent discovery. Every exported counter keeps its
source path. A mount-root limit must not be mistaken for the job's effective
limit; unrecorded parent cgroups can still impose additional limits. CPU quotas/throttling should
be interpreted together with the recorded membership and the scheduler allocation.
Cgroup counters are cumulative and sampled after the timed call: differences
between records cover the intervening unsampled calls as well.

Process resource deltas cover the sampled backend propagation, including worker
execution. They exclude input loading, manager preparation/materialization,
Minuit work and JSON/file collection. `previousSampleExportSeconds` separately
reports collection/serialization/write overhead of the preceding sample.
OS accounting and diagnostic counters still add some cost to sampled calls;
compare with a run without probes before using timings as a final benchmark.

### What this cannot prove

These probes do not directly measure DRAM bandwidth, cache misses, IPC or exact
NUMA remote-memory traffic. If the first reports point toward those limits, a
second pass using Linux hardware counters (for example `perf`, subject to cluster
permissions) will be needed. The initial probes work without those permissions.

## Validation

The Release build and `232CpuBackendProfiling.py` verify unchanged predictions,
cache accounting, sampling limits, live flushing and append behavior. The summary
script is exercised on the exported report. Existing CPU/MPS propagation and
feature tests are also run. Local macOS validation cannot validate Linux-only
resource fields or the CCLyon job environment; the first grid report is needed
for that part.

A local OA2024 Asimov fit (416,034 events, 3,529,068 unique dials,
6,561,717 event/dial references; four threads; post-fit Hesse skipped) completed
with 89 samples in a 516 KiB report. The most expensive sampled cache call
recomputed 477,862 spline responses, while many calls changed only inexpensive
normalizations. This illustrates why cache timing must be interpreted together
with the invalidated dial workload. Sample export on macOS had a median of about
0.122 ms and maximum of 0.422 ms; these figures exclude initialization/inventory
and worker instrumentation, and do not predict Linux export costs.

The first CCLyon report revealed a median export cost of 258 ms per sample,
with about 31 s spent exporting before the final sampled call (roughly 5.6% of
the observed elapsed time). Heavy system snapshots were therefore spaced out
as described above. The cause within system collection is not yet isolated;
new per-category read timers will distinguish process-file and cgroup reads.
The effect of this change on Linux still needs measurement. Existing kernel
stage timings exclude this export cost.
