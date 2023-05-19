![](./images/hpcLogo.png)

The HPC cluster (University of Geneva) provides a wide range of
software that can be linked using the `module` command.
The official documentation is available on the 
[UniGe website](https://doc.eresearch.unige.ch/hpc/applications_and_libraries).

## Environment Setup

The following chain of commands can be either set in the
user `.bash_profile` or in a dedicated bash script.

```sh
echo "Loading ROOT module"
ml load GCC/11.2.0
ml load OpenMPI/4.1.1
module load ROOT

echo "Loading yaml-cpp module"
ml load GCCcore/11.3.0
module load yaml-cpp/0.7.0

echo "Loading other modules..."
module load nlohmann_json

module load Python
module load CMake
module load git
module load CUDA
```

The versions are automatically chosen to match
the selected software.

Alright! You are now ready to compile :-).


## Building Gundam

As for the general guide we propose to compile the software with this tree
directory:

```sh
export WORK_DIR="$HOME"
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

```sh
mkdir -p $BUILD_DIR/gundam
mkdir -p $INSTALL_DIR/gundam
cd $BUILD_DIR/gundam
cmake \
  -D CMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR/gundam \
  -D CMAKE_BUILD_TYPE=Release \
  $REPO_DIR/gundam/.
make install -j4
```

Once the software is installed, let's link the `bin`
and `lib` folders:

```sh
export PATH="$INSTALL_DIR/gundam/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/gundam/lib:$LD_LIBRARY_PATH"
```

You can put these lines in your bash profile as well.

Now you should be able to run your first gundam command:

```sh
gundamFitter
```

Congrats! Now gundam is installed on your cluster! :-D



