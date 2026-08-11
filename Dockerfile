# Set the ROOT and Ubuntu versions used by the default build.
# ROOT image tags follow the form: <ROOT_VERSION>-ubuntu<UBUNTU_VERSION>.
# These values can be overridden with --build-arg, or ROOT_IMAGE can be
# provided directly for compatibility with older build commands.
ARG ROOT_VERSION=6.34.00
ARG UBUNTU_VERSION=24.04
ARG ROOT_IMAGE=rootproject/root:${ROOT_VERSION}-ubuntu${UBUNTU_VERSION}

FROM $ROOT_IMAGE AS base

# FROM rootproject/root:6.32.00-ubuntu24.04 as base
# FROM rootproject/root:6.30.06-ubuntu22.04 as base

SHELL ["/bin/bash", "-c"]

RUN apt-get update

ENV WORK_DIR /home/work
ENV REPO_DIR $WORK_DIR/repo
ENV BUILD_DIR $WORK_DIR/build
ENV INSTALL_DIR $WORK_DIR/install

RUN mkdir -p $REPO_DIR
RUN mkdir -p $BUILD_DIR
RUN mkdir -p $INSTALL_DIR

# Install the prerequisites.  Install individually and allow
# installation to fail since ubuntu tends to be a little jumpy about
# which packages are distributed.

RUN apt-get install -y git || true
RUN apt-get install -y libyaml-cpp-dev || true
RUN apt-get install -y nlohmann-json3-dev || true
RUN apt-get install -y libvdt-dev || true
RUN apt-get install -y python3-venv || true

# Copying GUNDAM source files
COPY ./src $REPO_DIR/src
# COPY ./submodules $REPO_DIR/submodules # submodules are not pulled on github
COPY ./cmake $REPO_DIR/cmake
COPY ./CMakeLists.txt $REPO_DIR/CMakeLists.txt
COPY ./.git $REPO_DIR/.git
COPY ./tests $REPO_DIR/tests

RUN python3 -m venv $REPO_DIR/venv
# Keep pybind11 independent of the Ubuntu package version.
RUN . $REPO_DIR/venv/bin/activate && \
    python -m pip install --upgrade pip && \
    python -m pip install pybind11==2.13.6 && \
    if [ -f $REPO_DIR/tests/requirements.txt ]; then python -m pip install -r $REPO_DIR/tests/requirements.txt; fi

# Checking out missing code
WORKDIR $REPO_DIR
RUN git submodule update --init --recursive

# Now build GUNDAM
WORKDIR $BUILD_DIR
RUN . $REPO_DIR/venv/bin/activate && \
    cmake \
      -D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -D WITH_PYTHON_INTERFACE=ON \
      -D pybind11_DIR="$(python -m pybind11 --cmakedir)" \
      $REPO_DIR 
RUN make -j3 install

# run the tests
RUN . $INSTALL_DIR/setup.sh && CTEST_OUTPUT_ON_FAILURE=1 make test

# activate it
ENV PATH=${INSTALL_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${INSTALL_DIR}/lib:${PYTHONPATH}

WORKDIR /home
# End of the file
