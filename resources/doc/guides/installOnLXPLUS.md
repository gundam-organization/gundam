![](./images/lxplusLogo.png)


This is a rough guide to get going on LXPlus. 
GUNDAM has pretty modern dependencies so we’ll need to pick these up
from what we can find on cvmfs etc.
LXPlus has recently switched towards "Red Hat Enterprise Linux release 9.4",
aka el9, which changed the dependencies paths.
This guide has been updated with the current software libraries available
on the platform.


## Common software available

Common software libraries can be found under `/cvmfs/sft.cern.ch/lcg/views/`.
They are documented on https://lcginfo.cern.ch/. 
To this date, the latest LCG release is 106 (Jun 19, 2024).
We can set up this environment by sourcing the dedicated bash file:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
```

For the rest of the guide we assume the software files are organised in 
three directories: `$REPO_DIR`, `$BUILD_DIR` and `$INSTALL_DIR`.

For instance in my configuration on LXPlus is:
```bash
export REPO_DIR=/eos/home-a/adblanch/software/repo
export BUILD_DIR=/eos/home-a/adblanch/software/build
export INSTALL_DIR=/eos/home-a/adblanch/software/install
```

You should create those directories before going any further in the guide:

```bash
mkdir -p $REPO_DIR
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
```


## GUNDAM Installation

### Clone the source code

First we need to clone the repository:

```bash
cd $REPO_DIR
git clone https://github.com/gundam-organization/gundam.git
cd gundam
```

Now need to check out a given version of GUNDAM and pull the associated submodules
(sub git repositories). Here is an example on how to pull the latest tagged version:

```bash
./update.sh --latest
```

Then create the dedicated build and install directory for GUNDAM:

```bash
mkdir $BUILD_DIR/gundam
mkdir $INSTALL_DIR/gundam
```

Now GUNDAM should be ready to build.


### Build the apps and libraries

On LXPlus, the CUDA libraries are available which means we can be enabled the support for
GPU.

```bash
cd $BUILD_DIR/gundam

# remove the cache is you already built
rm $BUILD_DIR/gundam/CMakeCache.txt

CLUSTER_OPTIONS="-D USE_STATIC_LINKS=ON"
CLUSTER_OPTIONS="$CLUSTER_OPTIONS -D WITH_CUDA_LIB=ON"
CLUSTER_OPTIONS="$CLUSTER_OPTIONS -D CMAKE_CUDA_COMPILER=/cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/bin/nvcc"
CLUSTER_OPTIONS="$CLUSTER_OPTIONS -D CMAKE_CUDA_ARCHITECTURES=all"

cmake \
      -D CMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR/gundam \
      -D CMAKE_BUILD_TYPE=RELEASE \
      $CLUSTER_OPTIONS \
      $REPO_DIR/gundam/.
      
make install
```


### Setup the environment

Once the make command succeeded, we have to make the gundam bin and lib directories 
referenced within the environment.
Put these lines in your `.bash_profile`:

```bash
echo "Setting up env for GUNDAM"
export PATH="$INSTALL_DIR/gundam/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/gundam/lib:$LD_LIBRARY_PATH"
```

```bash
source $HOME/.bash_profile
```

You should be able to run the gundam apps:

```sh
gundamFitter
```


```
11:59:42 [gundamFitter.cxx]: Usage:
11:59:42 [gundamFitter.cxx]: dry-run {--dry-run,-d}: Perform the full sequence of initialization, but don't do the actual fit. (trigger)
11:59:42 [gundamFitter.cxx]: config-file {-c,--config-file}: Specify path to the fitter config file (1 value expected)
11:59:42 [gundamFitter.cxx]: nb-threads {-t,--nb-threads}: Specify nb of parallel threads (1 value expected)
11:59:42 [gundamFitter.cxx]: output-file {-o,--out-file}: Specify the output file (1 value expected)

11:59:42 [gundamFitter.cxx]: Provided arguments:
11:59:42 [gundamFitter.cxx]: No options were set.
11:59:42 [gundamFitter.cxx]: Usage:
11:59:42 [gundamFitter.cxx]: dry-run {--dry-run,-d}: Perform the full sequence of initialization, but don't do the actual fit. (trigger)
11:59:42 [gundamFitter.cxx]: config-file {-c,--config-file}: Specify path to the fitter config file (1 value expected)
11:59:42 [gundamFitter.cxx]: nb-threads {-t,--nb-threads}: Specify nb of parallel threads (1 value expected)
11:59:42 [gundamFitter.cxx]: output-file {-o,--out-file}: Specify the output file (1 value expected)

11:59:42 [gundamFitter.cxx]: Provided arguments:
11:59:42 [gundamFitter.cxx]: No options were set.

11:59:42 [gundamFitter.cxx]: ��
terminate called after throwing an instance of 'std::runtime_error'
  what():  "config-file" option was not specified.
Aborted
```

Congrats! Now gundam is installed on your cluster! :-D
