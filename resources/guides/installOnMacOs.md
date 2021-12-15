![](./images/macOsLogo.png)

To compile on macOS you'll need to install XCode with the command line tools. Once it is done, 
dependencies on macOS can be handled by the package manager [Homebrew](https://brew.sh/index_fr):

```bash
brew install \
  root \
  yaml-cpp \
  nlohmann-json
```

### Compiling the code

Let's create the Build and Install folder:

```bash
mkdir -p $BUILD_DIR/gundam
mkdir -p $INSTALL_DIR/gundam
```

Now let's generate binaries:

```bash
cd $BUILD_DIR/gundam
cmake \
  -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR/gundam \
  -D CMAKE_BUILD_TYPE=Release \
  $REPO_DIR/gundam/.
make -j 4 install
```

If you did get there without error, congratulations! Now GUNDAM is installed on you machine :-D.

To access the executables from anywhere, you have to update you `$PATH` and `$LD_LIBRARY_PATH`
variables:

```bash
export PATH="$INSTALL_DIR/gundam/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/gundam/lib:$LD_LIBRARY_PATH"
```



