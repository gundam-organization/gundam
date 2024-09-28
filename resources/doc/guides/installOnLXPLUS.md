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



## Environment Setup
Go to the directory you cloned GUNDAM into and create the following script, filling in the correct location for REPO_DIR:
```
export REPO_DIR="/path/to/dir/above/GUNDAM"
export BUILD_DIR="$REPO_DIR/gundam/build"
export INSTALL_DIR="$REPO_DIR/gundam/install"

mkdir -p $BUILD_DIR
mkdir -p $REPO_DIR
mkdir -p $INSTALL_DIR

# enabling a C++11 compatible compiler (i.e. GCC5+)
source /cvmfs/sft.cern.ch/lcg/contrib/gcc/11/x86_64-centos7/setup.sh

# ROOT
export PATH="/afs/cern.ch/user/a/adblanch/public/software/install/root/6.26.10_x86_64_el7_gcc11_cxx17/bin:$PATH"
export LD_LIBRARY_PATH="/afs/cern.ch/user/a/adblanch/public/software/install/root/6.26.10_x86_64_el7_gcc11_cxx17/lib:$LD_LIBRARY_PATH"

# git with proper submodule support
export PATH="/cvmfs/sft.cern.ch/lcg/contrib/git/2.28.0/x86_64-centos7/bin:$PATH"
export LD_LIBRARY_PATH="/cvmfs/sft.cern.ch/lcg/contrib/git/2.28.0/x86_64-centos7/lib:$LD_LIBRARY_PATH"

# JSON (my own compiled version, couldn't find a suitable one)
export PATH="/afs/cern.ch/user/a/adblanch/public/software/install/json/bin:$PATH"
export LD_LIBRARY_PATH="/afs/cern.ch/user/a/adblanch/public/software/install/json/lib:$LD_LIBRARY_PATH"

# yaml-cpp
export PATH="/cvmfs/sft.cern.ch/lcg/releases/yamlcpp/0.6.3-d05b2/x86_64-centos7-gcc11-opt/bin:$PATH"
export LD_LIBRARY_PATH="/cvmfs/sft.cern.ch/lcg/releases/yamlcpp/0.6.3-d05b2/x86_64-centos7-gcc11-opt/lib:$LD_LIBRARY_PATH"

# zlib -> optional dependency of GUNDAM
export PATH="/cvmfs/sft.cern.ch/lcg/releases/zlib/1.2.11-8af4c/x86_64-centos7-gcc11-opt//bin:$PATH"
export LD_LIBRARY_PATH="/cvmfs/sft.cern.ch/lcg/releases/lzlib/1.2.11-8af4c/x86_64-centos7-gcc11-opt//lib:$LD_LIBRARY_PATH"

#setup the fitter (this won't work until we build it!):
source $BUILD_DIR/Linux/setup.sh
```

Then source the script you just made.

## Building Gundam

To build, do the following:

```
cd $BUILD_DIR
cmake3 ../
make -j4
make install
```

Now let's re-source the setup script you just made or just run the last line yourself: 

```
source $BUILD_DIR/Linux/setup.sh
```

Now you should be able to run your first gundam command:

```sh
gundamFitter
```

The program should stop while complaining that you didn't provide a config
file:

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
