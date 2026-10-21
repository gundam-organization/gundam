# Backends library

## From configuration to computation

The original engine connects the user's configuration to an in-memory model:
parameters, events, dials, samples and their relationships. Its object structure
expresses the requested physics and keeps the full range of features available.
It remains responsible for constructing and connecting these objects, whichever
execution backend is selected.

That organization also serves the original propagation path, but is not always
the most efficient representation for repeated numerical evaluation. Following
relationships between objects introduces indirection, and generic execution must
accommodate different dial types and optional behavior. These costs matter when
a fit repeats the same operations over millions of events and dial references.

Backends provide hardware-specific execution of this model. They arrange data
and operations for the target processor: compact arrays, reusable numerical
caches, specialized loops and an appropriate division of parallel work. The
goal is to prepare the execution structure once, then reuse it as parameter
values change throughout the fit.

The separation lets configuration and physics remain in the engine while each
backend chooses a memory layout and execution strategy suited to its hardware.
Shared numerical semantics keep these implementations consistent; a common
model does not require identical storage, scheduling or floating-point precision
on every device.

## Available backends

Implementation details, numerical precision and performance choices are documented
with each backend:

- [CPU structure](cpu/BackendStructure.md) and [optimization choices](cpu/ReweightOptimization.md).
- [MPS structure](mps/BackendStructure.md).

## Connection to the engine

`BackendManager` connects the likelihood interface to a backend: it prepares
runtime inputs, requests propagation and materializes the outputs needed by the
fitter or host-side consumers.

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

Each backend decides how to retain, pack or upload the model for its hardware.
Mutable caches and outputs belong to the executor; the common representation
keeps numerical definitions separate from those execution choices.

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

Cache invalidation follows the dependencies of the stored quantity. Reusing a
dial response does not imply that downstream event weights or histograms can be
reused. External-weight generations and observable revisions identify changes
that parameter values alone cannot describe; producers must report them through
the input interfaces. Each backend documents which intermediate results it caches.

Shared semantics define dial evaluation, input transformations, response bounds
and smoothing. Executors may specialize traversal, scheduling and memory layout
while preserving these numerical rules within their supported precision and
capabilities.
