# Metal buffers shared with an independent Python process

This first transport test validates native Metal -> Python/Torch MPS -> native
Metal access to the same mapped allocation on a unified-memory Mac. It is
generic: no nu-waves, oscillation parameters, ROOT, GUNDAM Python interface,
embedded interpreter or fitter configuration is involved.

The native producer is compiled against Foundation and Metal only. It launches
the selected Python executable with `posix_spawn`. Only the worker-side Torch
adapter is built using that Python's headers and packages. Changing the worker
venv does not require rebuilding GUNDAM against a different CPython runtime.

## Run

Requirements: macOS 15+, a unified-memory Metal device, Xcode command-line tools,
and a worker venv with Torch and Ninja. Torch 2.9.1 / CPython 3.14 was tested on
the local Apple GPU. The adapter uses native ATen APIs and must be retested when
upgrading Torch. Compilation happens in a temporary or explicitly selected
directory, never in the standard GUNDAM build.

From the repository, with the test runner's Python activated:

```bash
python tests/python/run.py --test 745BackendsMpsSharedBufferIpc.py -v -- \
    --python-executable /path/to/worker/venv/bin/python \
    --require-mps --build-directory /tmp/gundam-mps-ipc-build
```

The runner defaults the worker executable to its own `sys.executable`. The
worker may use a different environment. Each worker environment builds its own
adapter: use a separate build directory per Python/Torch environment.

Without `--require-mps`, a missing Torch package, unavailable MPS or unsupported
OS produces an explicit skip. Once dependencies are present, build errors,
protocol errors and numerical errors fail the test. MPS CPU fallback is disabled
inside the worker process. `--count` selects 1 to 1,000,000 events (default 65,539).

## What the test does

1. A standalone native executable creates a page-aligned shared file mapping and
   wraps it with `newBufferWithBytesNoCopy` / `MTLStorageModeShared`.
2. A separate Python process maps the same file, creates its own no-copy Metal
   buffer and exposes parameters, event inputs and weights as MPS tensor views.
   Offsets are nonzero for event arrays and output. Tensor deleters retain the
   mapping for the lifetime of the views.
3. A native GPU kernel writes parameters and event inputs. The script computes
   `weights = parameters[0] * inputs + parameters[1]` on MPS, directly into the
   shared output view.
4. A native GPU consumer verifies every weight and the two output guards. This
   repeats for eight generations, changing both parameters and inputs on GPU
   while retaining the same allocation and views.
5. Only the accumulated GPU error counter is read by the CPU, after all eight
   generations. An invalid mapping size is separately checked for rejection;
   successful shutdown must remove the backing file.

The protocol carries only setup metadata, generation numbers and acknowledgments.
No numerical input, parameter or output arrays travel through the pipes. Python
`.cpu()`, `.numpy()` and `.item()` calls are rejected during evaluation.

This demonstrates shared GPU access without explicit host/device array copies.
The shared backing memory remains host-addressable, as expected for Metal shared
storage on unified-memory hardware. It does not demonstrate exporting an existing
private Metal allocation or private VRAM on a discrete GPU.

## Synchronization and limits

For this first test, handoffs are blocking: native production completes before
the evaluate command; the worker synchronizes its MPS stream before acknowledging;
the native consumer finishes before the next generation. These are CPU waits,
not CPU data transfers. GPU event-based asynchronous handoff is future work.
The worker process has an explicitly checked PID, distinct from the producer.

This is a test transport, not yet a `PythonWorker` implementation. In particular,
the real worker's `run(command)` API, CUDA transport, general descriptors, worker
failure recovery and concurrent scheduling are not implemented here.

The agreed future configuration places memory selection inside `workerConfig`:

```yaml
options:
  type: PythonWorker
  workerConfig:
    pythonExecutable: /path/to/worker/venv/bin/python
    evalScript: ./reweight.py
    backendBufferMemoryLocation: onDevice  # Future option; default: onHost.
```

This option is not parsed by production GUNDAM yet. The prototype's input/script
interface should be agreed before connecting this transport to the real worker.
