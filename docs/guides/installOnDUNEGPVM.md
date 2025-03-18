This is the tutorial about gundam installation on FNAL DUNE gpvm.
## I. Simply lovely
For those who don’t want to waste time installing multiple packages on their dunegpvm, just follow this section.
### 0. Setup
```bash
source /exp/dune/app/users/flynnguo/spack_install/setup-env.sh
spack load gcc@12.2.0
spack load root@6.28.12
spack load yaml-cpp@0.7.0%gcc@12.2.0
spack load cmake@3.27.7%gcc@12.2.0
spack load jsoncpp@1.9.5%gcc@12.2.0
```
### 1. Shell setup 
```bash
cd /exp/dune/app/users/$USER
mkdir GUNDAM
cd GUNDAM 
export WORK_DIR="/exp/dune/app/users/$USER/GUNDAM"
export INSTALL_DIR="$WORK_DIR/Install/"
export BUILD_DIR="$WORK_DIR/Build/"
export REPO_DIR="$WORK_DIR/Repositories/"
```
If it's the first time you define those, don't forget to `mkdir`, create these path!
```bash
mkdir -p $INSTALL_DIR
mkdir -p $BUILD_DIR
mkdir -p $REPO_DIR
```
### 2. Cloning the source code
GUNDAM source code is officially available under the 
[GUNDAM-organization on GitHub](https://github.com/gundam-organization/gundam).
To copy the code on your computer or cluster, we recommend to use GIT.
We assume 
```bash
cd $REPO_DIR
git clone --recursive -b main https://github.com/gundam-organization/gundam.git
cd gundam
```
For GUNDAM users, it is recommended for you to check out the latest
tagged version of this repository. A simple bash script allows you to
check out the latest tagged version by tapping:
```bash
./update.sh --latest
```
### 3. Building the code
```bash
mkdir -p $BUILD_DIR/gundam
mkdir -p $INSTALL_DIR/gundam
cd $BUILD_DIR/gundam
cmake \
-D CMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR/gundam \
-D CMAKE_BUILD_TYPE=Release \
$REPO_DIR/gundam/.
make install -j$(nproc)         
```
### 4. Future login and Run the code 
Save these aliases in ```~/.bashrc```:
```bash
alias myspack='. /exp/dune/app/users/flynnguo/spack_install/setup-env.sh' # here do want to used pre-installed packages under Flynn's area
alias setgundam='source /exp/dune/app/users/$USER/GUNDAM/Install/gundam/setup.sh'
alias yamllib_gundam='export LD_LIBRARY_PATH=/exp/dune/app/users/flynnguo/spack_install/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_placeholde/yaml-cpp/0.7.0/linux-almalinux9-x86_64_v3-gcc-12.2.0-tkehcjd266veu3oj3hupgrbewiimoe4s/lib64:$LD_LIBRARY_PATH'
```

Exit and relogin:
```bash
source ~/.bashrc
myspack
setgundam
yamllib_gundam
gundamFitter
```
### 6. Update Gundam
If you updated any gundam codes, you need to rebuild and reinstall
```bash
cd $BUILD_DIR/gundam
cmake \
-D CMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR/gundam \
-D CMAKE_BUILD_TYPE=Release \
$REPO_DIR/gundam/.
make -j4
make install
```

## II. Create your own spack area
For those who want to learn how to install multiple packages on their dunegpvm, follow this section.
### 1. Setup
```bash
cd /exp/dune/app/users/$USER
source /cvmfs/larsoft.opensciencegrid.org/spack-packages/setup-env.sh
spack load fermi-spack-tools@main
git config --global --add safe.directory /cvmfs/larsoft.opensciencegrid.org/spack-packages//.git
make_subspack --with_padding /cvmfs/larsoft.opensciencegrid.org/spack-packages/ /exp/dune/app/users/$USER/spack_install
spack unload fermi-spack-tools@main
. spack_install/setup-env.sh
```
Now you can install the packages you want
```bash
spack list <package name>
spack -d install <package name>
spack find --paths <package name>
```
### 2. Next time login
```bash
cd /exp/dune/app/users/$USER
. spack_install/setup-env.sh
spack load <package name>
```
