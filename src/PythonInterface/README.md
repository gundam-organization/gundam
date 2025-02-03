# PyGundam - A python interface for GUNDAM

## Building

With the CMake command, you need to enable the compilation of the
python extension:

```bash
cd $BUILD_DIR/gundam
cmake -D WITH_PYTHON_INTERFACE=ON ./
```

You need to have `pybind11` installed on your computer. On macOS, you
can install it via:

```bash
brew install pybind11
```

## Setup

Make sure the lib folder of GUNDAM is listed under $PYTHONPATH:

```bash
export PYTHONPATH="$INSTALL_DIR/gundam/lib:$PYTHONPATH"
```

## Running

PRELIMINARY

Go to the OA input folder, and run:

```bash
python
Python 3.13.1 (main, Dec  3 2024, 17:59:52) [Clang 16.0.0 (clang-1600.0.26.4)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import PyGundam
>>> g = PyGundam.PyGundam("configOa2021.yaml")
```

