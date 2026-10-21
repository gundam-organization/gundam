# Backends library

The Backends library executes parameter propagation on CPU or device-specific
implementations. `BackendManager` connects the likelihood interface to a backend:
it prepares runtime inputs, requests propagation and materializes the outputs
needed by the fitter or host-side consumers.

## Model, inputs and outputs

`EngineLayout` separates a flat numerical model (`EngineView`) from bindings to
the original GUNDAM objects. The model contains dial descriptors and payloads,
event-to-dial indices, event-weight descriptors, observable/binning metadata and
likelihood definitions. It is built once for a fixed model layout.

At each propagation, the manager supplies parameter values, external-weight
buffer views and generations, and observable revisions or host values when
needed. Numerical executors consume these inputs without traversing the original
event and dial object graph. Changes to model structure require rebuilding the
layout; changes to parameter values do not.

The CPU backend retains shared ownership of the immutable model built by the
manager. Its standalone `build(const EngineView&)` API takes a snapshot. Other
backends can pack or upload the model according to their hardware requirements;
a common semantic representation does not require identical execution layouts.
Mutable caches and outputs belong to the executor.

A propagation token identifies its outputs. Readiness and materialization requests
control which event weights, histograms, observable state and likelihood results
are available to the caller, including any necessary device-to-host transfer.

## Cache responsibilities

| Data | Purpose and lifetime |
|---|---|
| Event/dial associations | Record which unique dials affect each event, preserving their product order; retained with the model. |
| Dial responses | Reuse eligible numerical responses until their effective inputs change; managed by the executor. |
| Observable/binning state | Track relevant parameter, external-buffer and observable revisions so dependent state is updated when needed. |
| Numerical outputs | Publish results for a propagation token; retained storage can be overwritten by later evaluations. |

Cached dial responses do not memoize an entire likelihood evaluation. In the CPU
backend, unchanged parameters can avoid response evaluation while event weights
and histograms are still recomputed. External-weight generations and observable
revisions are part of invalidation where those inputs affect observable state;
producers must report changes through the existing input interfaces.

Shared semantics define dial evaluation, input transformations, response bounds
and smoothing. Executors may specialize traversal, scheduling and memory layout
while preserving these numerical rules. CPU calculations use double precision;
device precision and execution capabilities are backend-specific.

See [CPU backend structure](cpu/CpuBackend.md) for data ownership and worker
distribution, and [CPU reweight optimization](cpu/ReweightOptimization.md) for
cache policies, optimization choices and benchmark findings.
