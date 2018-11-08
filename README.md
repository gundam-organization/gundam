# Super-xsllhFitter

## Introduction

The goal of the Super-xsllhFitter is to provide a general purpose likelihood-based fit framework for performing cross section analyses. The current state of the code and documentation is very much work in progress and is really not ready for general use. However, if you're desperate or can't wait then read on!

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

Tagged versions of the code can also be downloaded from the Tags section on the GitLab page.

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

The default build is `DEBUG`, which compiles the libraries statically and includes debugging symbols. The other build type is `RELEASE`, which can be enabled by either calling cmake with `-DCMAKE_BUILD_TYPE=RELEASE` or by using the ccmake command. The `RELEASE` build enables compiler optimizations, disables debug symbols, and builds/links the libraries as shared objects.

For future use, the root setup.sh script will perform all the necessary setup to run the fitter. Once configured with CMake, only the `make install` step needs to be performed if the code needs to be rebuilt/recompiled.

## Running the Code

Good luck.
