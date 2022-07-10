![](./images/cc_in2p3_logo.png)

As several libraries are available to the users, compiling on the
cluster can be challenging. Here we described how to get the software
up and running for job submissions. We assumed you are already part of
the T2K computing group, i.e. you can access the SPS storage: `ls /sps/t2k/`.

## Environment Setup

The following bash commands can be set in a single bash file you can readAndFill
each time you login.

Let's first make sure we have a not-too-old compiler. Here an example with 
GCC-7:

```sh
ccenv gcc 7.3.0
source /opt/rh/devtoolset-7/enable
export CC="$(which gcc)"
export CXX="$(which g++)"
echo "   ├─ GCC Version : $(gcc --version | head -n 1)"
```

Note that you should be able to compile with a more recent version of GCC.
Our goal is to keep the software compatible with the most recent version of
GCC. If you find any issue related to the compilation, please post a ticket
on the issue tracker of github!

Let's now use a more up-to-date version of cmake:
```sh
ccenv cmake 3.20.2
```

You can also provide a specific version with `ccenv`. Available versions can
be shown with `ccenv cmake --list`.

Since all the available versions of ROOT on the CC are compiled either with a quite
old compiler or with missing features, a shared ROOT library has been installed
in T2K folder `/sps/t2k/common/software`. As we also need the `yaml-cpp` library
we advise to source the  dedicated env file:

```sh
source /sps/t2k/common/software/env.sh
```

The output should look like this:
```sh
Setting up T2K libs...
Linking libs in /sps/t2k/common/software/install
   ├─ Adding : nano-5.9
   ├─ Adding : root-v6-24-06
   ├─ Adding : yaml-cpp
Selecting ROOT v6-24-06...
   ├─ ROOT Prefix : /sps/t2k/common/software/install/root-v6-24-06
   ├─ ROOT Version : 6.24/06
T2K common software has been setup.
```

Alright! You are now ready to compile :-).


## Building Gundam

As for the general guide we propose to compile the software with this tree
directory:

```sh
export WORK_DIR="/sps/t2k/<YOURUSERNAME>/work"
export INSTALL_DIR="$WORK_DIR/install/"
export BUILD_DIR="$WORK_DIR/build/"
export REPO_DIR="$WORK_DIR/repositories/"
```

Then create these path:

```sh
mkdir -p $INSTALL_DIR
mkdir -p $BUILD_DIR
mkdir -p $REPO_DIR
```

Let's clone Gundam with git:

```sh
cd $REPO_DIR
git clone https://github.com/nadrino/gundam.git
cd $REPO_DIR/gundam
# fetch the latest tagged version
git checkout $(git describe --tags `git rev-list --tags --max-count=1`)
# don't forget to initialize the submodules
git submodule update --init --recursive
```

Now let's build the thing. In the cmake command, we explicitly specify where
to find `yaml-cpp`:

```sh
mkdir -p $BUILD_DIR/gundam
mkdir -p $INSTALL_DIR/gundam
cd $BUILD_DIR/gundam
cmake \
  -D CMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR/gundam \
  -D CMAKE_BUILD_TYPE=Release \
  -D YAMLCPP_DIR=/sps/t2k/common/software/install/yaml-cpp \
  $REPO_DIR/gundam/.
make install -j4
```

In principle in should be compiling fine... Or this guide is failing to do
his job!! Please report any issue!

Once the software is installed, let's link the `bin` and `lib` folders:

```sh
export PATH="$INSTALL_DIR/gundam/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/gundam/lib:$LD_LIBRARY_PATH"
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



