## Installation

There are several requirements for building the fitter:

* GCC 4.8.5+ or Clang 3.3+ (a C++11 enabled compiler)
* CMake 3.5+
* ROOT 5 or 6

ROOT needs either Minuit or Minuit2 and optionally the MathMore package enabled to perform the minimization. The recommendation is to have both Minuit and Minuit2 enabled. In addition it is highly recommended to have a working OpenMP installation to take advantage of parallelism when running the code.

To checkout a tagged version of the code using git:

```bash
$ git clone https://gitlab.com/cuddandr/xsLLhFitter.git
$ git checkout -b <choose a branch name> <tag>
```

Tagged versions of the code can also be downloaded as zipped archives from the Tags section on the GitLab page.

Set up the ROOT environment before attempting to build by:

```bash
$ source /path/to/ROOT/bin/thisroot.sh
```

Then source the package setup script.

```bash
$ source /path/to/xsLLhFitter/setup.sh
```

The first time this script is run it will notify you that it cannot find the build setup script, this is normal. The fitter is designed to be built in a build directory specified by the user and is configured using CMake.

To build (with default settings):

```bash
$ mkdir build; cd build
$ cmake ../
$ make install -j
$ source $(uname)/setup.sh
```

The default build is `DEBUG`, which compiles the libraries statically and includes debugging symbols. The other build type is `RELEASE`, which can be enabled by either calling cmake with `-DCMAKE_BUILD_TYPE=RELEASE` or by using the ccmake command. The `RELEASE` build enables compiler optimizations, disables debug symbols, and builds/links the libraries as shared objects. Other options can be passed to CMake by using `-DOPTION_NAME=VALUE` when invoking cmake, or by using ccmake.

There are a few extra options to configure the build. The default option is listed in brackets:

* CMAKE_CXX_EXTENSIONS [OFF]: Enable GNU extensions to C++ language (-std=gnu++11)
* CXX_MARCH_FLAG [OFF]: Enable cpu architecture specific optimizations
* CXX_WARNINGS [ON]: Enable most C++ warning flags
* COLOR_OUTPUT [ON]: Enable colored terminal output

For future use, the root setup.sh script will perform all the necessary setup to run the fitter. Once configured with CMake, only the `make install` step needs to be performed if the code needs to be rebuilt/recompiled. Sometimes `make clean` will need to be run before `make install` to correctly build.

---

## Errors and Warnings

This is an incomplete list of known errors and warnings how they might potentially be solved

### CMake Finding Incorrect Compiler

CMake may not find the correct compiler when there are multiple available. If this is the case, set the compiler manually by either passing the `-DCMAKE_CXX_COMPILER` and `-DCMAKE_C_COMPILER` variables when configuring or exporting the `CC` and `CXX` variables before configuring. All previous build files will need to be deleted for cmake to reconfigure properly. For example:

```bash
$ cmake -DCMAKE_CXX_COMPILER=/path/to/compiler/executable -DCMAKE_C_COMPILER=/path/to/compiler/executable ../
```

Or by exporting shell variables:

```bash
$ export CC=/path/to/compiler/executable
$ export CXX=/path/to/compiler/executable
$ cmake ..
```
### Illegal Instruction

When running or compiling the code an `illegal instruction` error may appear. This can be caused by running on a different computer architecture than the code was compiled for when using the `CXX_MARCH_FLAG` option. Run `make clean` in the build directory and then `make install` to rebuild the code, which should fix the issue. Alternatively, turn off the `CXX_MARCH_FLAG` if enabled and rebuild.

---

## Running the Code

The Super-xsllh Fitter is built to be a framework of tools which are designed to work together. There are the tools that produce inputs in the correct format, the main fit program, the error propagation, and a number of scripts which may or may not be useful. In addition, there is a set of programs designed to run T2KReWeight and produce splines which is currently not included in this repository.

Good luck.
