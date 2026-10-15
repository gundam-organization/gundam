# External Weight Dials

[< back to parent (Dial)](Dial.md)

External weight dials let GUNDAM delegate the evaluation of event weights to an external worker. `PythonWorker` is the first implementation; the common worker interface is also intended for precompiled external libraries.

This is useful when the response depends on many events at once, for example an oscillation engine that can evaluate all event weights in a vectorized CPU/GPU backend.

The external worker is persistent. GUNDAM loads the requested event inputs once, then each likelihood propagation only sends the current fit parameters and asks the worker to update the event weights.

## YAML Configuration

The dial is configured in a `dialSetDefinitions` entry:

```yaml
parameterSetList:
  - name: "OscillationParameters"
    isEnabled: true
    nominalStepSize: 0.1

    dialSetDefinitions:
      - dialType: ExternalWeight
        applyOnDataSets: [ "MySample" ]
        dialInputList:
          - name: "deltaMsq"
          - name: "sinSqTheta"
        options:
          type: "PythonWorker"
          useBinnedWeights: false
          inputEventVarList:
            - "[Enu]"
            - "[FlavorEmit]"
            - "[FlavorDetect]"
          workerConfig:
            pythonExecutable: "/path/to/venv/bin/python"
            evalScript: "./config/externalWeightWorker.py"
            scriptArgs: [ "--model-config", "./config/osc.yaml" ]

    parameterDefinitions:
      - name: "deltaMsq"
        priorValue: 2.5e-3
        priorType: Flat
        parameterLimits: [0.0, 0.01]
      - name: "sinSqTheta"
        priorValue: 0.5
        priorType: Flat
        parameterLimits: [0.0, 1.0]
```

### Options

| Option                        | Type       | Description                                                     | Default |
|-------------------------------|------------|-----------------------------------------------------------------|---------|
| dialType                      | string     | Must be `ExternalWeight`                                        |         |
| dialInputList                 | list(json) | Fit parameters passed to the worker                             |         |
| options.type                  | string     | Worker implementation; currently `PythonWorker`                 |         |
| options.inputEventVarList     | list(str)  | Event variables made available to every worker                  | empty   |
| options.useBinnedWeights      | bool       | Evaluate one weight per configured bin instead of per event      | false   |
| options.binning                | json       | `BinSet` configuration used when `useBinnedWeights` is enabled  |         |
| options.workerConfig          | json       | Configuration specific to the selected worker                   |         |
| workerConfig.pythonExecutable | string     | Python executable used to start `PythonWorker`                  |         |
| workerConfig.evalScript       | string     | Python worker script                                            |         |
| workerConfig.scriptArgs       | list(str)  | Extra command-line arguments passed to the Python worker script | empty   |
| workerConfig.initScript       | string     | Currently ignored by the shared-memory Python worker            |         |

`inputEventVarList` entries use the same variable naming as other event formulas. In practice, prefer bracket notation for variables from the dataset `variableDict`:

```yaml
options:
  inputEventVarList: [ "[Enu]", "[FlavorEmit]", "[FlavorDetect]" ]
```

`inputEventVarList` is optional. Omit it when the weight depends only on fit parameters. Each configured input is evaluated while loading the dataset and stored in memory by the common worker layer.

When `useBinnedWeights` is enabled, `options.binning` uses the standard GUNDAM
`BinSet` syntax. GUNDAM creates one dispatcher per bin and `EventDialCache`
associates each event with the matching dispatcher. The Python worker receives
one input value per bin, evaluated at the bin center, and writes one weight
per bin. Binning variables are automatically added to `inputEventVarList`.
Every input variable requested by the worker must be defined in every bin.
Events outside the configured bins do not receive the external dial.

For example:

```yaml
options:
  type: PythonWorker
  useBinnedWeights: true
  binning:
    binningDefinition:
      - name: Enu
        nBins: 100
        min: 0.0
        max: 10.0
      - name: FlavorEmit
        values: [12, 14]
      - name: FlavorDetect
        values: [12, 14]
  workerConfig:
    pythonExecutable: "/path/to/venv/bin/python"
    evalScript: "./config/externalWeightWorker.py"
```

In binned mode, `command_["nBins"]` is the size of the NumPy arrays exposed
through `command_["inputs"]` and `command_["weights"]`.

At the moment, the shared-memory transport exposes event inputs as `float64` arrays. If an input is semantically an integer, such as a flavor code, the worker should cast it explicitly before using it.

## Python Worker Contract

The worker is launched as:

```bash
/path/to/python externalWeightWorker.py --worker
```

If `scriptArgs` is provided, GUNDAM inserts those arguments before `--worker`. For example:

```bash
/path/to/python externalWeightWorker.py --model-config ./config/osc.yaml --worker
```

Before starting the persistent worker, GUNDAM runs a light preflight check with:

```bash
/path/to/python -m py_compile externalWeightWorker.py
```

This does not accelerate the fit in a meaningful way, but it catches Python syntax errors early and fails before the likelihood initialization continues.

GUNDAM communicates with the worker through small JSON commands on stdin/stdout. The numerical arrays are not transferred through JSON. They are exposed as POSIX shared-memory buffers.

The worker must:

- read one JSON command per line from stdin,
- write exactly one JSON response per command to stdout,
- reserve stdout for JSON responses only,
- write logs/debug messages to stderr if needed,
- return `{"status": "ok"}` for successful commands.

### Commands

`initialize` is sent once, before the first evaluation. It contains:

```json
{
  "command": "initialize",
  "nEvents": 10,
  "inputs": {
    "Enu": {
      "shmName": "gdmEW_12345_i0",
      "dtype": "float64",
      "shape": [10]
    },
    "FlavorEmit": {
      "shmName": "gdmEW_12345_i1",
      "dtype": "float64",
      "shape": [10]
    },
    "FlavorDetect": {
      "shmName": "gdmEW_12345_i2",
      "dtype": "float64",
      "shape": [10]
    }
  },
  "parameters": [
    { "name": "deltaMsq", "title": "#0_deltaMsq", "index": 0 },
    { "name": "sinSqTheta", "title": "#1_sinSqTheta", "index": 1 }
  ],
  "parameterBuffer": {
    "shmName": "gdmEW_12345_parameters",
    "dtype": "float64",
    "shape": [2]
  },
  "weights": {
    "shmName": "gdmEW_12345_w",
    "dtype": "float64",
    "shape": [10]
  }
}
```

`evaluate` is sent at each propagation where the fit parameters changed:

```json
{ "command": "evaluate" }
```

The worker should read the current parameter values from `parameterBuffer`, compute one weight per event, write the result into `weights`, then respond:

```json
{ "status": "ok" }
```

`shutdown` is sent when GUNDAM destroys the worker:

```json
{ "command": "shutdown" }
```

## Minimal Worker Example

GUNDAM preloads the worker protocol, including JSON parsing, shared-memory
attachment and response handling. A new-style script only needs a `run`
function. On every call, `command_` contains:

- `command_['command']`: `initialize`, `evaluate` or `shutdown`;
- `command_['inputs']`: a dictionary of NumPy arrays;
- `command_['parameters']`: a dictionary mapping parameter names to values;
- `command_['weights']`: the writable NumPy output array.

The `initialize` command also contains `parameterInfo`, with the original
parameter metadata. The function may return a response dictionary; when it
returns `None`, GUNDAM sends `{"status": "ok"}`. An optional `configure`
function receives `scriptArgs` before the first command, allowing the script
to parse its command-line arguments.

```python
import sys

import numpy as np


def configure(arguments_):
    # arguments_ contains workerConfig.scriptArgs. For example, with:
    #   scriptArgs: ["--baseline-km", "295"]
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-km", type=float, default=295.0)
    configure.options = parser.parse_args(arguments_)


def run(command_):
    command_name = command_["command"]

    if command_name == "initialize":
        # Inputs are already attached NumPy arrays at this point. The
        # parameterInfo list contains name/title/index metadata.
        print("Loaded parameters:", command_["parameterInfo"], file=sys.stderr)
        return {"status": "ok"}

    if command_name == "evaluate":
        parameters = command_["parameters"]
        enu = command_["inputs"]["Enu"]
        flavor_emit = np.rint(command_["inputs"]["FlavorEmit"]).astype(np.int64)
        flavor_detect = np.rint(command_["inputs"]["FlavorDetect"]).astype(np.int64)

        phase = 1.267 * parameters["deltaMsq"] * configure.options.baseline_km / enu
        transition_prob = parameters["sinSqTheta"] * np.sin(phase) ** 2
        command_["weights"][:] = np.where(
            flavor_emit == flavor_detect,
            1.0 - transition_prob,
            transition_prob,
        )
        return {"status": "ok"}

    if command_name == "shutdown":
        # GUNDAM closes the shared-memory handles after run() returns.
        return {"status": "ok"}

    return {"status": "error", "message": "unknown command: " + command_name}
```

`attach_array`, `respond`, `run_worker` and the `__main__` block are no
longer needed. GUNDAM calls `configure` once, then calls `run` for each
command and handles the shared-memory cleanup after `shutdown`.

## Internal Engine Flow

In the default mode, GUNDAM creates one `ExternalWeightDispatcher` per
selected event. In `useBinnedWeights` mode, it creates one dispatcher per
configured bin instead; `EventDialCache` associates each selected event with
the corresponding bin dispatcher.

After all events are loaded:

1. GUNDAM allocates one shared-memory `double` array per requested input variable (none when `inputEventVarList` is omitted).
2. GUNDAM fills those input arrays once from the selected events, or with bin-center values in binned mode.
3. GUNDAM allocates one shared-memory `double` array for the current fit parameters.
4. GUNDAM allocates one shared-memory `double` array for the output weights. Its size is `nEvents` in the default mode and `nBins` in binned mode.
5. GUNDAM starts the Python worker and sends the shared-memory names and array metadata in the `initialize` JSON command.

At each propagation:

1. GUNDAM writes the current fit parameter values into `parameterBuffer`.
2. GUNDAM sends the small JSON command `{ "command": "evaluate" }`.
3. The Python worker computes all event or bin weights and writes them into the shared `weights` array.
4. The common worker layer updates the shared weight vector.
5. Each `ExternalWeightDispatcher` returns the weight corresponding to its event or bin index.

The large numerical arrays are therefore never serialized to JSON and are not written to disk. JSON is only used as a small control protocol for metadata and commands.

The current cache-manager backend does not support `ExternalWeight` dials. When such dials are present, GUNDAM falls back to the CPU propagation path.
