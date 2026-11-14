#!/usr/bin/env python3
"""Optional two-process Metal/Torch no-copy test, independent of the fitter."""

import argparse
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", default=sys.executable,
                        help="Worker venv Python. GUNDAM's native test does not link CPython.")
    parser.add_argument("--require-mps", action="store_true",
                        help="Fail rather than skip missing Torch or Metal.")
    parser.add_argument("--build-directory", type=Path)
    parser.add_argument("--count", type=int, default=65539)
    args = parser.parse_args()
    if not 1 <= args.count <= 1000000:
        parser.error("--count must be between 1 and 1000000")
    reason = None
    if sys.platform != "darwin" or int(platform.mac_ver()[0].split(".")[0]) < 15:
        reason = "This test requires macOS 15+ and a unified-memory Metal device"
    native = Path(__file__).resolve().parents[2] / "mps"
    worker = native / "MpsSharedBufferWorker.py"
    if reason is None:
        probe = subprocess.run([args.python_executable, "-B", str(worker), "--probe"],
                               text=True, capture_output=True, timeout=30)
        if probe.returncode == 77:
            reason = probe.stdout.strip()
        elif probe.returncode:
            raise RuntimeError(probe.stdout + probe.stderr)
        else:
            print(probe.stdout.strip(), flush=True)
    if reason:
        if args.require_mps:
            raise RuntimeError(reason)
        print("SKIP:", reason)
        return 0

    buildDirectory = (args.build_directory or Path(tempfile.mkdtemp(prefix="gundam-mps-ipc-build-"))).resolve()
    buildDirectory.mkdir(parents=True, exist_ok=True)
    executable = buildDirectory / "MpsSharedBufferProducer"
    # This compilation has no Python headers, libpython, Torch or ROOT.
    subprocess.run(["clang++", "-std=c++17", "-O2", "-fno-objc-arc",
                    str(native / "MpsSharedBuffer.mm"), str(native / "MpsSharedBufferProducer.mm"),
                    "-framework", "Foundation", "-framework", "Metal", "-o", str(executable)],
                   check=True, timeout=60)
    subprocess.run([args.python_executable, "-B", str(worker), "--prepare", str(buildDirectory)],
                   check=True, timeout=180)
    # Exercise bad descriptors without a producer or GPU work.
    with tempfile.TemporaryDirectory(prefix="gundam-mps-ipc-invalid-") as directory:
        invalid = Path(directory) / "wrong-size"
        invalid.write_bytes(b"invalid")
        code = (
            "import sys, torch; sys.path.insert(0, sys.argv[1]); "
            "import gundam_mps_shared_buffer as ipc; "
            "ipc.attach(sys.argv[2], 17)"
        )
        result = subprocess.run([args.python_executable, "-c", code, str(buildDirectory), str(invalid)],
                                capture_output=True, text=True, timeout=30)
        if result.returncode == 0 or "Unexpected shared buffer size" not in result.stderr:
            raise AssertionError("Invalid descriptor was not rejected: " + result.stderr)
    with tempfile.TemporaryDirectory(prefix="gundam-mps-ipc-data-") as directory:
        mapping = Path(directory) / "buffers"
        subprocess.run([str(executable), args.python_executable, str(worker), str(buildDirectory),
                        str(mapping), str(args.count)], check=True, timeout=60)
        if mapping.exists():
            raise AssertionError("Producer left its shared allocation behind")
    print("PASS: invalid descriptor rejected and shared allocation cleaned up", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
