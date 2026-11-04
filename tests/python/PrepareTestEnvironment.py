"""Prepare the build-local CTest environment using the selected interface Python."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import venv


def main():
    directory = Path(sys.argv[1])
    python = directory / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    requirements = Path(__file__).with_name('requirements.txt')
    # Refresh the environment if the selected interpreter or requirements change.
    signature = hashlib.sha256(requirements.read_bytes() +
                               (sys.executable + sys.version).encode()).hexdigest()
    stamp = directory / '.gundam-requirements'
    if python.exists() and stamp.exists() and stamp.read_text() == signature:
        print('Reusing Python test environment:', directory, flush=True)
        return

    print('Preparing Python test environment:', directory, flush=True)
    # Preserve packages supplied by the selected interpreter (e.g. pybind11/ROOT).
    # Never install test dependencies into the system interpreter itself.
    venv.EnvBuilder(with_pip=True, system_site_packages=True, clear=True).create(directory)
    subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(requirements)], check=True)
    stamp.write_text(signature)


if __name__ == '__main__':
    main()
