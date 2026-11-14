"""Test worker using the selected venv; no nu-waves or GUNDAM Python dependency."""

import os
from pathlib import Path
import sys

# Set before importing Torch. Unsupported operators must fail, never fall back.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"


def main():
    if sys.argv[1] == "--probe":
        try:
            import torch
            if not torch.backends.mps.is_available():
                print("MPS is unavailable")
                return 77
        except ImportError as error:
            print(str(error))
            return 77
        print("Worker Python:", sys.executable, "Torch:", torch.__version__)
        return 0

    import torch
    if sys.argv[1] == "--prepare":
        from torch.utils.cpp_extension import load
        native = Path(__file__).resolve().parent
        buildDirectory = Path(sys.argv[2])
        buildDirectory.mkdir(parents=True, exist_ok=True)
        load(name="gundam_mps_shared_buffer",
             sources=[str(native / "MpsSharedBuffer.mm"), str(native / "MpsSharedBufferWorker.mm")],
             extra_cflags=["-O2", "-fno-objc-arc"],
             extra_ldflags=["-framework", "Foundation", "-framework", "Metal",
                            "-framework", "MetalPerformanceShaders", "-framework", "MetalPerformanceShadersGraph"],
             build_directory=str(buildDirectory), verbose=True)
        return 0

    if sys.argv[1] != "--worker":
        raise ValueError("Expected --probe, --prepare or --worker")
    from unittest.mock import patch

    sys.path.insert(0, sys.argv[2])
    import gundam_mps_shared_buffer
    parameters, inputs, weights = gundam_mps_shared_buffer.attach(sys.argv[3], int(sys.argv[4]))
    if any(value.device.type != "mps" for value in (parameters, inputs, weights)):
        raise AssertionError("Imported buffers are not MPS tensors")
    print(f"ready {os.getpid()}", flush=True)
    nextGeneration = 0
    for line in sys.stdin:
        parts = line.split()
        if parts == ["shutdown"]:
            print("bye", flush=True)
            return 0
        if parts != ["evaluate", str(nextGeneration)]:
            raise ValueError("Unexpected command or stale generation")
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("CPU read")), \
             patch.object(torch.Tensor, "numpy", side_effect=AssertionError("NumPy conversion")), \
             patch.object(torch.Tensor, "item", side_effect=AssertionError("Scalar read")):
            with torch.no_grad():
                torch.mul(inputs, parameters[0], out=weights)
                weights.add_(parameters[1])
        # Completion handshake, not a data transfer. The native producer waits
        # before issuing its command; the consumer starts only after this reply.
        torch.mps.synchronize()
        print(f"ok {nextGeneration}", flush=True)
        nextGeneration += 1
    raise RuntimeError("Unexpected producer EOF")


if __name__ == "__main__":
    sys.exit(main())
