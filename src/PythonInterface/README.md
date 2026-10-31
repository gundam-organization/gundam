# A python interface for GUNDAM

## Building

You need to have `pybind11` installed on your computer. On macOS, you
can install it via:

```bash
brew install pybind11 # make sure you're using the right `python`
```

With the CMake command, you need to enable the compilation of the
python extension:

```bash
cd $BUILD_DIR/gundam
cmake -D WITH_PYTHON_INTERFACE=ON ./
```

On certain computing clusters, CMake won't be able to find the `pybind11` library. You can do so with the following
option:

```
-D WITH_PYTHON_INTERFACE=ON -D pybind11_DIR=/path/to/pybind11/cmake/files
```


With CMake 3.18 or newer, `GUNDAM_PYTHON_VERSIONS=AUTO` (the default)
searches `PATH` for `python3` and `python3.x` executables and builds the
three newest usable CPython major/minor versions (or all if fewer are available).
One module is built per version; aliases and virtual environments
with the same version are deduplicated. The first usable installation
in `PATH` wins. Interpreters without matching development files are
skipped with a configure message; configuration fails if none are usable.
The pybind11 installation must support the selected Python versions.

Explicit selections have no three-version limit. To select versions explicitly:

```bash
cmake -S . -B build -DWITH_PYTHON_INTERFACE=ON \
  '-DGUNDAM_PYTHON_VERSIONS=3.13;3.14'
```

Absolute interpreter paths can also be supplied in this list. Explicit
selections fail configuration if an interpreter or its development files
are missing. Use this option instead of `Python_EXECUTABLE` to select a
single interpreter. Re-run CMake after adding a Python installation to `PATH`.

The `GUNDAM` build target builds all selected interfaces. Individual targets
are named `GUNDAM_python3_13`, `GUNDAM_python3_14`, etc. The core C++
libraries are shared between them. Installed modules coexist in `lib` with
ABI-specific names such as `GUNDAM.cpython-313-darwin.so`; each Python
imports its matching module using `import GUNDAM` and the same `PYTHONPATH`.


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
>>> import GUNDAM
>>> GUNDAM.setLightOutputMode(True)
>>> GUNDAM.setNumberOfThreads(2)
...
```

Example for an Asymov fit

```python
import GUNDAM
GUNDAM.setRuntimeWorkingDirectory("/path/to/run/directory")
GUNDAM.setLightOutputMode(True)
GUNDAM.setNumberOfThreads(2)

cb = GUNDAM.ConfigUtils.ConfigBuilder("config.yaml")
cr = GUNDAM.ConfigUtils.ConfigReader(cb.getConfig())
cr.defineField(GUNDAM.ConfigUtils.ConfigReader.FieldDefinition("fitterEngineConfig"))
fitterEngineConfig = cr.fetchValueConfigReader("fitterEngineConfig")

e = GUNDAM.FitterEngine()
e.setConfig(fitterEngineConfig)
e.configure()

e.getLikelihoodInterface().setForceAsimovData(True)
e.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)

e.initialize()

e.fit()

# Draw a new parameter point around the post-fit best fit.
# This only updates the GUNDAM parameter values; propagate/evaluate explicitly
# if you need refreshed histograms or likelihood values.
minimizer = e.getMinimizer()
minimizer.throwPostfitParameters()
e.getLikelihoodInterface().propagateAndEvalLikelihood()
```

The likelihood data type can be selected with
`LikelihoodInterface.DataType`. The available values are `Asimov`, `Toy`, and
`RealData`:

```python
likelihood = e.getLikelihoodInterface()

likelihood.setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
likelihood.setDataType(GUNDAM.LikelihoodInterface.DataType.Toy)
likelihood.setDataType(GUNDAM.LikelihoodInterface.DataType.RealData)
```

Relative paths used by GUNDAM are resolved from the current process working
directory. When using the Python interface, call
`GUNDAM.setRuntimeWorkingDirectory(...)` before loading relative config files or
initializing the runtime. Pass `createIfMissing=True` to create the directory if
needed:

```python
GUNDAM.setRuntimeWorkingDirectory("/path/to/run/directory", createIfMissing=True)
print(GUNDAM.getRuntimeWorkingDirectory())
```
Relative paths used by GUNDAM are resolved from the current process working
directory. When using the Python interface, call
`GUNDAM.setRuntimeWorkingDirectory(...)` before loading relative config files or
initializing the runtime. Pass `createIfMissing=True` to create the directory if
needed:

```python
GUNDAM.setRuntimeWorkingDirectory("/path/to/run/directory", createIfMissing=True)
print(GUNDAM.getRuntimeWorkingDirectory())
```
