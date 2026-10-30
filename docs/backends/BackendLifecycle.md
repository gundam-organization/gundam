# Backend configuration and lifecycle

The manager keeps one configured, empty instance for each compiled backend and
builds a model for only the active backend. `CPU` is always available; CUDA
requires a CUDA build, and MPS requires an Apple build. Device creation and
model allocation happen during activation, so a compiled backend can still
fail to activate if no suitable hardware is available.

Without CUDA support, no source or header from `Implementations/CUDA` is
required or compiled, and its include directory is not exported. The common
factory reports the compiled backend names; the manager does not include a
specialized backend header. CUDA diagnostics in Python and CUDA-specific tests
are built only when CUDA support is enabled.

```yaml
fitterEngineConfig:
  likelihoodInterfaceConfig:
    backendManagerConfig:
      isEnabled: true
      useBackend: CPU
      options:
        - name: CPU
          weightScheduling: Dynamic
        - name: CUDA
          deviceIndex: 0
          propagationMode: cachedResponses
          likelihoodBlockSize: 0
        - name: MPS
          propagationMode: cachedResponses
```

With `isEnabled: true`, `useBackend` defaults to `CPU`. With `isEnabled: false`
(the default), the manager starts without an active backend and can be activated
later from Python. Options for a known backend absent from the build are ignored,
including its backend-specific fields. Explicitly selecting that backend still
throws. Unknown names and duplicate option entries throw.

`type` remains a deprecated alias of `useBackend`. The old top-level
`backendConfig` field and other backend-specific fields beside `options` are
rejected. Move those settings into the corresponding named entry.

```python
manager = engine.getBackendsManager()  # after engine.initialize()
manager.getAvailableBackendNames()    # e.g. ["CPU", "MPS"]
manager.getActiveBackendName()        # "CPU", "CUDA", "MPS", or ""
manager.deactivateBackend()
manager.activate("CPU")               # use stored options
manager.activate("MPS", GUNDAM.JsonType('{"propagationMode":"recomputeResponses"}'))
manager.deactivateBackend()
manager.activate("MPS")               # retains the last successful override
engine.evaluateLikelihood()
```

An explicit JSON configuration replaces that backend's stored options, using
backend defaults for omitted fields. Invalid names/configurations fail before
the current backend is released. A hardware/model-build failure leaves the
manager inactive and releases any partially allocated model; activation raises
an error. `replaceBackend(name, config)` remains an alias for explicit activation.

Deactivation waits for pending work, materializes available event state, weights,
histograms and statistical likelihood, releases the active backend's resources,
and invalidates its result tokens. It is safe to call repeatedly. With no active
backend, `engine.evaluateLikelihood()` uses the original propagation engine.
Activation rebuilds the backend model from the already-loaded events and dials;
it never reopens dataset files. Observable responses, dynamic event migration,
and smoothed bin contributions are supported. The initialized likelihood's prior
bin mask is preserved across rebuilds. Sample lists and likelihood bin counts
must remain unchanged.

The state transfer happens when switching backends, even with automatic
materialization disabled. It is outside fit evaluations: supported resident
likelihoods still read back only the scalar statistical likelihood during normal
evaluations, plus the existing backend control metadata. Backend changes must
be made between evaluations, rather than concurrently with a fit thread.

`tests/fast-tests/240BackendLifecycle.py` checks CPU/MPS/original-engine switches
with observable responses, explicitly dynamic and smoothed samples, weights,
histogram errors, Poisson and Barlow-Beeston 2022. It deletes the ROOT input
after loading to verify transitions do not require disk access. Set
`GUNDAM_TEST_CUDA=1` in a CUDA-enabled GPU allocation to include CUDA transitions.

The V100 benchmark defaults to `--execution-mode resident`: inputs are loaded
once per likelihood for all block-size/reuse/backend combinations, and backend switches reuse that data.
`--execution-mode isolated` remains available for independent process comparisons.
Model setup, backend changes and warmup are excluded from evaluation timings.
