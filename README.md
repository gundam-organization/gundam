# Super-xsllhFitter

## Introduction

The goal of the Super-xsllhFitter is to provide a general purpose likelihood-based fit framework for performing cross section analyses. The current state of the code and documentation is very much work in progress and is really not ready for general use. However, if you're desperate or can't wait then read on! There is no guarantee the fitter will be able to perform your analysis.

The code is under very active development and to give some kind of stability, it is recommended you checkout/download a tagged version of the fitter.

This document is currently all about the code for the fitter. For anything related to the principles behind the fitter, browse any of the following technotes: TN214, TN261, TN263, TN287, TN337, TN338.

## Installation

There are several requirements for building the fitter:
- GCC 4.8.5+ or Clang 3.3+ (a C++11 enabled compiler)
- CMake 3.5+
- ROOT 5 or 6

To checkout a tagged version of the code using git:

    $ git clone https://gitlab.com/cuddandr/xsLLhFitter.git
    $ git checkout -b <choose a branch name> <tag>

Tagged versions of the code can also be downloaded as zipped archives from the Tags section on the GitLab page.

Set up the ROOT environment before attempting to build by:

    $ source /path/to/ROOT/bin/thisroot.sh

Then source the package setup script.

    $ source /path/to/xsLLhFitter/setup.sh

The first time this script is run it will notify you that it cannot find the build setup script, this is normal. The fitter is designed to be built in a build directory specified by the user and is configured using CMake.

To build (with default settings):

    $ mkdir build; cd build
    $ cmake ../
    $ make install -j
    $ source $(uname)/setup.sh

The default build is `DEBUG`, which compiles the libraries statically and includes debugging symbols. The other build type is `RELEASE`, which can be enabled by either calling cmake with `-DCMAKE_BUILD_TYPE=RELEASE` or by using the ccmake command. The `RELEASE` build enables compiler optimizations, disables debug symbols, and builds/links the libraries as shared objects. Other options can be passed to CMake by using `-DOPTION_NAME` when invoking cmake, or by using ccmake.

There are a few extra options to configure the build. The default option is listed in brackets:
- CMAKE_CXX_EXTENSIONS [OFF]: Enable GNU extensions to C++ language (-std=gnu++11)
- CXX_MARCH_FLAG [OFF]: Enable cpu architecture specific optimizations
- CXX_WARNINGS [ON]: Enable most C++ warning flags
- COLOR_OUTPUT [ON]: Enable colored terminal output

### CMake Finding Incorrect Compiler

CMake may not find the correct compiler when there are multiple available. If this is the case, set the compiler manually by either passing the `-DCMAKE_CXX_COMPILER` and `-DCMAKE_C_COMPILER` variables when configuring or exporting the `CC` and `CXX` variables before configuring. All previous build files will need to be deleted for cmake to reconfigure properly. For example:

    $ cmake -DCMAKE_CXX_COMPILER=/path/to/compiler/executable -DCMAKE_C_COMPILER=/path/to/compiler/executable ../

Or by exporting shell variables:

    $ export CC=/path/to/compiler/executable
    $ export CXX=/path/to/compiler/executable
    $ cmake ..

For future use, the root setup.sh script will perform all the necessary setup to run the fitter. Once configured with CMake, only the `make install` step needs to be performed if the code needs to be rebuilt/recompiled.

## Running the Code

Good luck.
