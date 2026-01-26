## Environment Set Up

Set up your environment:

```sh
ml load gcc/11.2.0
ml load root/6.26
```

## Building Gundam

As for the general guide we propose to compile the software with this tree
directory:

```sh
export WORK_DIR="$HOME/Work"
export INSTALL_DIR="$WORK_DIR/Install/"
export BUILD_DIR="$WORK_DIR/Build/"
export REPO_DIR="$WORK_DIR/Repositories/"
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
git clone --recurse-submodules https://github.com/gundam-organization/gundam.git
cd gundam
git submodule sync
git submodule update --init
```

Specify where to find `yaml-cpp`:

```sh
export PATH="/gpfs/projects/McGrewGroup/uyevarou/install/yaml_install:$PATH";
export LD_LIBRARY_PATH="/gpfs/projects/McGrewGroup/uyevarou/install/yaml_install/lib64/:$LD_LIBRARY_PATH";
```

Now let's build the thing:

```sh
mkdir -p $BUILD_DIR/gundam
mkdir -p $INSTALL_DIR/gundam
cd Build/gundam/

cmake \
-D CMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR/gundam \
-D CMAKE_BUILD_TYPE=Release \
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



