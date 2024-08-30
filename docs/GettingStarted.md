---
layout: default
title: Getting started 
next_page: "https://ulyevarou.github.io/GUNDAM-documentation/usage.html"
---

# Installation instructions

## Prerequisites

There are several requirements for building the fitter:

- CMake 3.12+
- A C++14 enabled compiler
  - Recommended GCC 8+ ( GCC 5 minumim )
  - Recommended Clang 9+ ( Clang 3.4 minimum )
- [ROOT 6, compiled with C++14 or later](https://github.com/root-project/root)
- [JSON for Modern C++](https://github.com/nlohmann/json)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)



## Shell setup

In this guide, it is assumed you have already defined the following bash environment
variables:

- `$REPO_DIR`: the path to the folder where your git projects are stored. This guide
  will download this repo into the subdirectory `$REPO_DIR/gundam`.

- `$BUILD_DIR`: the path where the binaries are built. As for the previous variables,
  this guide will work under `$BUILD_DIR/gundam`.

- `$INSTALL_DIR`: the path where the binaries are installed and used by the shell.
  Same here: this guide will work under `$INSTALL_DIR/gundam`.

As an example, here is how I personally define those variables. This script is executed
in the `$HOME/.bash_profile` on macOS or `$HOME/.bashrc` on Linux, as they can be used
for other projects as well.

```bash
export WORK_DIR="$HOME/Documents/Work"
export INSTALL_DIR="$WORK_DIR/Install/"
export BUILD_DIR="$WORK_DIR/Build/"
export REPO_DIR="$WORK_DIR/Repositories/"
```

If it's the first time you define those, don't forget to `mkdir`!

```bash
mkdir -p $INSTALL_DIR
mkdir -p $BUILD_DIR
mkdir -p $REPO_DIR
```

## Cloning the source code

GUNDAM source code is officially available under the 
[GUNDAM-organization on GitHub](https://github.com/gundam-organization/gundam).
To copy the code on your computer or cluster, we recommend to use GIT.
We assume 

```bash
cd $REPO_DIR
git clone --recurse-submodules https://github.com/gundam-organization/gundam.git
cd gundam
```

For GUNDAM users, it is recommended for you to check out the latest
tagged version of this repository. A simple bash script allows you to
check out the latest tagged version by tapping:

```bash
./update.sh --latest
```

Alternatively GUNDAM users can choos the branch

```bash
git branch master
git pull origin selectedbranch
git submodule sync
git submodule update --init
```

Note that this command will also automatically check out the submodule
included in the project. Therefore, in order to update your code when
a new release is available, simply use the same command. Note that git versions 
before 2.0 may have issues to checkout the correct submodules (see issue #429)

## Building the code

```bash
cd $BUILD_DIR/gundam
cmake \
-D CMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR/gundam \
-D CMAKE_BUILD_TYPE=Release \
$REPO_DIR/gundam/.
make install -j$(nproc)
```

## Add GUNDAM to the PATH

```bash
export PATH="$INSTALL_DIR/gundam/bin:$PATH";
export LD_LIBRARY_PATH="$INSTALL_DIR/gundam/lib:$LD_LIBRARY_PATH";

```

# Installation for:

- [Compiling on macOS ](guides/installOnMacOs.md)
- [Compiling on CCLyon ](guides/installOnCCLyon.md)
- [Compiling on HPC cluster (University of Geneva)](guides/installOnHpc.md)
- [Compiling on LXPLUS](guides/installOnLXPLUS.md)
- Compiling on HPC cluster (Stony Brook University)
- Compiling on Cedar (Digital Research Alliance of Canada)

# Alternative installation procedure

```bash
cd $REPO_DIR/gundam
cd cmake/scripts
./gundam-setup.sh
./gundam-build.sh
```

this will create the build directory `gundam-${compiler}_${compiler_version}-${compiler_machine}`.
# Errors and warnings