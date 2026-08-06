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

```python
#!/usr/bin/env python3

import json
import sys
from multiprocessing import shared_memory

import numpy as np


def attach_array(description):
    # GUNDAM creates the POSIX shared-memory object and sends its name plus
    # the array metadata in the JSON command. Python only attaches to the
    # existing object; no numerical payload is copied here.
    try:
        # Python 3.13+ supports track=False. This tells Python's resource
        # tracker that the shared-memory lifetime is managed by GUNDAM, not
        # by this worker process.
        shm = shared_memory.SharedMemory(name=description["shmName"], track=False)
    except TypeError:
        # Older Python versions do not have track=False. They register the
        # block in the resource tracker by default, so unregister it manually
        # to avoid Python unlinking a buffer owned by GUNDAM at shutdown.
        shm = shared_memory.SharedMemory(name=description["shmName"])
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass

    # Build a NumPy view directly on top of the shared-memory buffer. This is
    # zero-copy: reading/writing this array reads/writes the same memory that
    # GUNDAM mapped on the C++ side.
    array = np.ndarray(tuple(description["shape"]), dtype=np.float64, buffer=shm.buf)

    # Keep both objects alive. The NumPy array views shm.buf, and the
    # SharedMemory handle must stay open as long as the array is used.
    return shm, array


def respond(payload):
    print(json.dumps(payload), flush=True)


def run_worker():
    state = {"shared_memory": []}

    for line in sys.stdin:
        command = json.loads(line)

        if command["command"] == "initialize":
            state["inputs"] = {}

            # Event-level input arrays are read-only from the worker point of
            # view. GUNDAM fills them once while loading the dataset.
            for name, description in command["inputs"].items():
                shm, array = attach_array(description)
                state["shared_memory"].append(shm)
                state["inputs"][name] = array

            # This small array is overwritten by GUNDAM before every
            # evaluate command. Its order matches command["parameters"].
            shm, array = attach_array(command["parameterBuffer"])
            state["shared_memory"].append(shm)
            state["parameters"] = array
            state["parameter_names"] = [entry["name"] for entry in command["parameters"]]

            # This output array must be filled by the worker with exactly one
            # weight per selected event.
            shm, array = attach_array(command["weights"])
            state["shared_memory"].append(shm)
            state["weights"] = array

            respond({"status": "ok"})

        elif command["command"] == "evaluate":
            parameters = dict(zip(state["parameter_names"], state["parameters"]))
            enu = state["inputs"]["Enu"]
            flavor_emit = np.rint(state["inputs"]["FlavorEmit"]).astype(np.int64)
            flavor_detect = np.rint(state["inputs"]["FlavorDetect"]).astype(np.int64)

            delta_msq = parameters["deltaMsq"]
            sin_sq_theta = parameters["sinSqTheta"]
            baseline_km = 295.0

            # Write in-place into the shared output buffer. GUNDAM will read
            # these values after receiving the status response.
            phase = 1.267 * delta_msq * baseline_km / enu
            transition_prob = sin_sq_theta * np.sin(phase) ** 2
            survival_prob = 1.0 - transition_prob
            same_flavor = flavor_emit == flavor_detect
            state["weights"][:] = np.where(same_flavor, survival_prob, transition_prob)
            respond({"status": "ok"})

        elif command["command"] == "shutdown":
            respond({"status": "ok"})
            for shm in state["shared_memory"]:
                # Close the Python handle only. The actual unlink is owned by
                # GUNDAM, which created the shared-memory object.
                shm.close()
            return 0

        else:
            respond({"status": "error", "message": "unknown command"})
            return 1

    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        sys.exit(run_worker())
    sys.exit(1)
```

## Internal Engine Flow

During dataset loading, GUNDAM creates one `ExternalWeightDispatcher` per selected event. Each dispatcher stores its event index and a shared reference to the weight vector owned by `ExternalWeightWorker`.

After all events are loaded:

1. GUNDAM allocates one shared-memory `double` array per requested input variable (none when `inputEventVarList` is omitted).
2. GUNDAM fills those input arrays once from the selected events.
3. GUNDAM allocates one shared-memory `double` array for the current fit parameters.
4. GUNDAM allocates one shared-memory `double` array for the output event weights.
5. GUNDAM starts the Python worker and sends the shared-memory names and array metadata in the `initialize` JSON command.

At each propagation:

1. GUNDAM writes the current fit parameter values into `parameterBuffer`.
2. GUNDAM sends the small JSON command `{ "command": "evaluate" }`.
3. The Python worker computes all event weights and writes them into the shared `weights` array.
4. The common worker layer copies the shared `weights` array into its weight vector.
5. Each `ExternalWeightDispatcher` returns the weight corresponding to its event index.

The large numerical arrays are therefore never serialized to JSON and are not written to disk. JSON is only used as a small control protocol for metadata and commands.

The current cache-manager backend does not support `ExternalWeight` dials. When such dials are present, GUNDAM falls back to the CPU propagation path.
