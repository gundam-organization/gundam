FROM rootproject/root as base

RUN apt-get dist-upgrade -y 
RUN apt-get update && apt-get upgrade -y
RUN apt-get install git libyaml-cpp-dev -y

ENV WORK_DIR /home/work
RUN mkdir -p $WORK_DIR
WORKDIR $WORK_DIR

ENV REPO_DIR $WORK_DIR/repo
ENV BUILD_DIR $WORK_DIR/build
ENV INSTALL_DIR $WORK_DIR/install

RUN mkdir -p $REPO_DIR
RUN mkdir -p $BUILD_DIR
RUN mkdir -p $INSTALL_DIR

SHELL ["/bin/bash", "-c"]

RUN mkdir -p $REPO_DIR/gundam
RUN mkdir -p $BUILD_DIR/gundam
COPY . $REPO_DIR/gundam

# sudo is required by github actions since git clone is done by root
RUN cd $REPO_DIR/gundam && \
    git submodule update --init --recursive && \
    cd $BUILD_DIR/gundam && \
    # for some reason yaml-cpp in not found by cmake, so put the paths manually
    cmake \
      -D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
#      -D WITH_CUDA=ON \
      $REPO_DIR/gundam && \
    make -j3 install && \
    . $INSTALL_DIR/setup.sh && \
    CTEST_OUTPUT_ON_FAILURE=1 make test

# End of the file
