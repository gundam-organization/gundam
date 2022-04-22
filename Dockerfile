FROM rootproject/root as base

RUN apt install git libyaml-cpp-dev -y

## install prerequests
#RUN sudo apt-get update && \
#    sudo apt-get install build-essential curl cmake file git ruby-full locales -y
#
## install ROOT & yaml-cpp
#RUN brew update && brew install root  && \
#    brew install yaml-cpp

ENV WORK_DIR /home/work
RUN mkdir -p $WORK_DIR
WORKDIR $WORK_DIR

ENV REPO_DIR $WORK_DIR/repo
ENV BUILD_DIR $WORK_DIR/build
ENV INSTALL_DIR $WORK_DIR/install

RUN mkdir -p $REPO_DIR
RUN mkdir -p $BUILD_DIR
RUN mkdir -p $INSTALL_DIR


#FROM base as gundumBuild
#COPY --from=base $WORK_DIR $WORK_DIR

SHELL ["/bin/bash", "-c"]

RUN mkdir -p $REPO_DIR/gundam
RUN mkdir -p $BUILD_DIR/gundam
COPY . $REPO_DIR/gundam

# sudo is required by github actions since git clone is done by root
RUN cd $REPO_DIR/gundam && \
    git submodule update --init --recursive && \
    cd $BUILD_DIR/gundam && \
    # for some reason yaml-cpp in not found by cmake, so put the paths manually
    cmake -D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        $REPO_DIR/gundam && \
    make -j6 install
