# Set which version of ROOT to be built with.  This is usually going
# to be latest, but there are sometimes problems.  The available tags
# are found at https://hub.docker.com/r/rootproject/root.  There can
# be only one!

# ROOT generally distributes docker for the latest ubuntu LTS release
# let the option to change this with `--build-arg ROOT_IMAGE=rootproject/root:6.24.06-ubuntu20.04`
# by default, we use:
ARG ROOT_IMAGE=rootproject/root:latest

FROM $ROOT_IMAGE as base

# FROM rootproject/root:6.32.00-ubuntu24.04 as base
# FROM rootproject/root:6.30.06-ubuntu22.04 as base

SHELL ["/bin/bash", "-c"]

RUN apt-get update

# Install the prerequisites.  Install individually and allow
# installation to fail since ubuntu tends to be a little jumpy about
# which packages are distributed.

RUN apt-get install -y git || true
RUN apt-get install -y libyaml-cpp-dev || true
RUN apt-get install -y nlohmann-json3-dev || true
RUN apt-get install -y libvdt-dev || true

ENV WORK_DIR /home/work
ENV REPO_DIR $WORK_DIR/repo
ENV BUILD_DIR $WORK_DIR/build
ENV INSTALL_DIR $WORK_DIR/install

RUN mkdir -p $REPO_DIR
RUN mkdir -p $BUILD_DIR
RUN mkdir -p $INSTALL_DIR

# Copying GUNDAM source files
COPY ./src $REPO_DIR/src
# COPY ./submodules $REPO_DIR/submodules # submodules are not pulled on github
COPY ./cmake $REPO_DIR/cmake
COPY ./CMakeLists.txt $REPO_DIR/CMakeLists.txt
COPY ./tests $REPO_DIR/tests
COPY ./.git $REPO_DIR/.git

# Checking out missing code
WORKDIR $REPO_DIR
RUN git submodule update --init --recursive

# Now build GUNDAM
WORKDIR $BUILD_DIR
RUN cmake \
      -D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -D ENABLE_GOOGLE_TESTS=ON \
      $REPO_DIR 
RUN make -j3 install

# run the tests
RUN . $INSTALL_DIR/setup.sh && CTEST_OUTPUT_ON_FAILURE=1 make test

# End of the file
