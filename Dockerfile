FROM rootproject/root as base

RUN apt-get dist-upgrade -y 
RUN apt-get update && apt-get upgrade -y
RUN apt-get install git libyaml-cpp-dev nlohmann-json3-dev -y

ENV WORK_DIR /home/work
ENV REPO_DIR $WORK_DIR/repo
ENV BUILD_DIR $WORK_DIR/build
ENV INSTALL_DIR $WORK_DIR/install

RUN mkdir -p $REPO_DIR
RUN mkdir -p $BUILD_DIR
RUN mkdir -p $INSTALL_DIR

SHELL ["/bin/bash", "-c"]

COPY . $REPO_DIR/.

COPY ./src $REPO_DIR/.
COPY ./submodules $REPO_DIR/.
COPY ./cmake $REPO_DIR/.
COPY ./CMakeLists.txt $REPO_DIR/.
COPY ./.git $REPO_DIR/.

WORKDIR $BUILD_DIR

# sudo is required by github actions since git clone is done by root
RUN git submodule update --init --recursive
RUN cmake \
      -D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -D ENABLE_CUDA=ON \
      $REPO_DIR
RUN make -j3 install

# setup env
RUN . $INSTALL_DIR/setup.sh

# run the tests
RUN CTEST_OUTPUT_ON_FAILURE=1 make test

# End of the file
