# GUNDAM — 風をあつめて

![GUNDAM banner](./resources/images/README/title/title.001.png)

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/gundam-organization/gundam/docker-image.yml)
[![GitHub forks](https://badgen.net/github/forks/gundam-organization/gundam/)](https://github.com/gundam-organization/gundam/network/members)
[![GitHub release](https://img.shields.io/github/release/gundam-organization/gundam.svg)](https://github.com/gundam-organization/gundam/releases/)
[![Generic badge](https://img.shields.io/badge/Users-Example_of_inputs-GREEN.svg)](https://github.com/gundam-organization/gundam-input-tutorial)

GUNDAM, standing for *Generalized and Unified Neutrino Data Analysis
Methods*, is a suite of applications which aims at performing various
statistical analysis with different purposes and setups.  It has been
developed as a fork of
[xsllhFitter](https://gitlab.com/cuddandr/xsLLhFitter), in the context of
the Upgrade of ND280 for the T2K neutrino experiment.  The GUNDAM source
code is officially available at the
[GUNDAM-organization on GitHub](https://github.com/gundam-organization/gundam).

The applications are configurable with a set of YAML/JSON files (see the
[configuration](./docs/configuration/TopLevel.md) documetation) so users do
not need to modify the code for each new study.  A lot of time and efforts
are usually invested by various working groups to debug and optimize pieces
of codes doing generic tasks, and GUNDAM lets that work be distributed to
all users. As GUNDAM is designed for maximize flexibility to accommodate
various physics fitting needs and makes sure that optimizations and
debugging benefit all analyses.

![Dependencies banner](resources/images/README/sections/sections.001.png)

### There are several requirements for building the fitter:

- CMake 3.12+
- A C++14 enabled compiler (C++17 is preferred)
  - Recommended GCC 8+ ( GCC 5 minimum )
  - Recommended Clang 9+ ( Clang 3.4 minimum )
- [ROOT 6, compiled with C++14 or later](https://github.com/root-project/root)
- [JSON for Modern C++](https://github.com/nlohmann/json)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)


### Building and Using

GUNDAM uses a standard git and cmake setup. This section will assume that you are putting your files in:

- `$REPO_DIR`: the path to the folder where your git projects are stored. This guide
  will download this repo into the subdirectory `$REPO_DIR/gundam`.

- `$BUILD_DIR`: the path where the binaries are built. As for the previous variables,
  this guide will work under `$BUILD_DIR/gundam`.

- `$INSTALL_DIR`: the path where the binaries are installed and used by the shell.
  Same here: this guide will work under `$INSTALL_DIR/gundam`.

#### Generic user setup.

Once GUNDAM is compiled and installed, the current installation can be configured using
```bash
source ${INSTALL_DIR}/gundam/setup.sh
```
This will setup the environment for GUNDAM, including the version of ROOT that was used to compile and install the project.

#### Development setup

A setup optimized for development in a bash shell is provided with the
GUNDAM cmake build system. This is optimized for working on several
parallel versions of gundam, possibly using different compilers, and
processors.  It keeps the installation and build directories inside the top
level repository with machine/system specific names.  The naming scheme is
`gundam-${compiler}_${compiler_version}-${compiler_machine}
```bash
cd ${REPO_DIR}/gundam
source ${ROOTSYS}/bin/thisroot.sh
source ./cmake/scripts/gundam-setup.sh
```
This will define the following environment variables.

- GUNDAM_ROOT   : The root directory for the repository.
- GUNDAM_TARGET : The machine specific target for the current compilation

The local behavior can be customized by setting environment variables before sourcing the setup script.

- GUNDAM_BUILD   : Force the location for the build
- GUNDAM_INSTALL : Set the installation directory (defaults: ${GUNDAM_BUILD})
- GUNDAM_JOBS    : Number of jobs to use during build
- GUNDAM_CMAKE_DEFINES -- Location specific cmake arguments

After setting up using `gundam-setup.sh`, two commands will be added to your environment

- gundam-build [help] : Compile, install, configure, test GUNDAM
- gundam-setup [extra] : Redo the environment setup. The `extra` argument is added to the `GUNDAM_TARGET` directory name.

The `gundam-build` command provides a standardized, location invariant, interface to the build system.  It can be run from any place and will recompile a new version into the gundam build directory.

#### Manual Shell setup

GUNDAM uses a standard cmake build system, so it can be configured by hand.

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

![Clone & Update banner](resources/images/README/sections/sections.002.png)

### Cloning and updating the source code

GUNDAM source code is officially available under the
[GUNDAM-organization on GitHub](https://github.com/gundam-organization/gundam).
To copy the code on your computer or cluster, we recommend to use GIT.

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

Note that this command will also automatically check out the submodule
included in the project. Therefore, in order to update your code when
a new release is available, simply use the same command. Note that git versions
before 2.0 may have issues to checkout the correct submodules (see issue #429).  If you run the `update.sh` script without any options, it will provide a help message.  Useful variants are

- `./update.sh --up` : Update to the head of the current branch.  Make sure all of the submodules are also updated.
- `./update.sh --head` : Checkout the main branch.  Make sure all of the submodules are also updated.

![Install banner](resources/images/README/sections/sections.003.png)

#### Compiling on macOS:

[![](docs/guides/images/macOsLogo.png)](docs/guides/installOnMacOs.md)


#### Compiling on CCLyon:

[![](docs/guides/images/cc_in2p3_logo.png)](docs/guides/installOnCCLyon.md)


#### Compiling on HPC:

[![](docs/guides/images/hpcLogo.png)](docs/guides/installOnHpc.md)


#### Compiling on LXPLUS:

[![](docs/guides/images/lxplusLogo.png)](docs/guides/installOnLXPLUS.md)



![Inputs banner](resources/images/README/sections/sections.004.png)

## Input examples

- Official input examples: [link to repo](https://github.com/gundam-organization/gundam-input-tutorial/tree/main)


## Documentation

- Gundam [User Documentation](https://gundam-organization.github.io/gundam/) is growing daily.  You can also find examples and other interesting materials under the [resources](resources/) hierarchy in this repository.

![Developers banner](resources/images/README/sections/sections.005.png)

### Development policy


#### Editing the code

- The `main` branch is the official HEAD of GUNDAM.
- Developers might make a [fork](https://github.com/gundam-organization/gundam/fork) of the `main` branch on their own GitHub account.
- Development should happen in a dedicated branch with a descriptive name of
  the feature you are developing. We recommend to tag your branches this way:
  - `fix/myFix`: for addressing specific issues with the code.
  - `feature/myFeature`: for adding specific feature.
  - `doc/myDoc`: for documentation additions.
  - `experimental/myBranch`: for your own implementation tests.  Note
      experimental branch should not be merged into `main`, so you are free
      to commit and implement whatever you want in those branches.  Those
      are just placeholders for you to identify which `feature` should be
      implemented.  After you figure out a fancy new idea, create a
      `feature` branch and cherry-pick the required changes.
- Commit messages must be explicit.
- Commit content must contain a few modifications to the code.
- Make your `feature` and `fix` branches short-lived with one new feature
  or fix per branch.  Fix branches should reference a specific issue that
  has been documented on by the gundam [issue
  tracker](https://github.com/gundam-organization/gundam/issues)

#### Merging to the official repository

- First of all, create a dedicated entry on the [Issue tracking page](https://github.com/gundam-organization/gundam/issues).
- Create a pull request (PR) of the branch from your fork into `main`.
- Copy-paste the associated issue link in the comment section of the PR.
- All the CI tests must be successful before merging.


#### Licence and rights

- Usage of the forked code is regulated by the code license.
- Share of the code is regulated by the code license.


![Lineage & Legacy banner](resources/images/README/sections/sections.006.png)

GUNDAM was born as a fork of the *xsllhFitter* project which was developped and used by
the cross-section working group of T2K. The original project can be found on *gitlab*:
[https://gitlab.com/cuddandr/xsLLhFitter](https://gitlab.com/cuddandr/xsLLhFitter).

GUNDAM has originally been developed as an new fitter to perform T2K
oscillation analysis, and provide an expandable base on which future
studies with the *Upgraded ND280 Detectors* will be performed.
