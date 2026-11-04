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
RUN apt-get install -y python3-pip || true

# Copying GUNDAM source files
COPY ./src $REPO_DIR/src
# COPY ./submodules $REPO_DIR/submodules # submodules are not pulled on github
COPY ./cmake $REPO_DIR/cmake
COPY ./CMakeLists.txt $REPO_DIR/CMakeLists.txt
COPY ./.git $REPO_DIR/.git
COPY ./tests $REPO_DIR/tests

# Prepare the selected Python environment before configuring GUNDAM.
RUN python3 -m venv $REPO_DIR/venv
RUN $REPO_DIR/venv/bin/python -m pip install --upgrade pip && \
    $REPO_DIR/venv/bin/python -m pip install pybind11==2.13.6 && \
    $REPO_DIR/venv/bin/python -m pip install -r $REPO_DIR/tests/python/requirements.txt

# Fetch the pinned C++ sources before CMake, which will use these local copies.
ENV DEPS_DIR=$WORK_DIR/dependencies
RUN git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git $DEPS_DIR/json && \
    git clone --depth 1 --branch yaml-cpp-0.9.0 https://github.com/jbeder/yaml-cpp.git $DEPS_DIR/yaml-cpp && \
    git clone --depth 1 --branch v1.16.0 https://github.com/google/googletest.git $DEPS_DIR/googletest

# Checking out missing code
WORKDIR $REPO_DIR
RUN git submodule update --init --recursive

# Now build GUNDAM
WORKDIR $BUILD_DIR
RUN --network=none . $REPO_DIR/venv/bin/activate && \
    cmake \
      -D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -D WITH_PYTHON_INTERFACE=ON \
      -D WITH_PYTHON_TEST_VENV=OFF \
      -D GUNDAM_PYTHON_VERSIONS=$REPO_DIR/venv/bin/python \
      -D FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=$DEPS_DIR/json \
      -D FETCHCONTENT_SOURCE_DIR_YAML-CPP=$DEPS_DIR/yaml-cpp \
      -D FETCHCONTENT_SOURCE_DIR_GTEST=$DEPS_DIR/googletest \
      -D FETCHCONTENT_FULLY_DISCONNECTED=ON \
      -D pybind11_DIR="$(python -m pybind11 --cmakedir)" \
      $REPO_DIR 
RUN --network=none make -j3 install

# run the tests
RUN --network=none . $INSTALL_DIR/setup.sh && CTEST_OUTPUT_ON_FAILURE=1 make test

# activate it
ENV PATH=${INSTALL_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${INSTALL_DIR}/lib:${PYTHONPATH}

WORKDIR /home
# End of the file
