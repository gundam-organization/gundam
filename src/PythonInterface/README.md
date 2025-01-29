# PyGundam - A python interface for GUNDAM

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

