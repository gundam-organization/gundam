FROM homebrew/ubuntu20.04 as base

# install prerequests
RUN sudo apt-get update && \
    sudo apt-get install build-essential curl cmake file git ruby-full locales -y

# install ROOT & yaml-cpp
RUN brew install root  && \
    brew install yaml-cpp

ENV GUNDIR /home/linuxbrew/gundam
RUN mkdir $GUNDIR
WORKDIR $GUNDIR

FROM base as gundum_build
COPY --from=base $GUNDIR $GUNDIR
ENV ROOTSYS /home/linuxbrew/.linuxbrew

COPY . $GUNDIR/

# sudo is required by github actions since git clone is done by root
RUN cd $GUNDIR && \
    source /home/linuxbrew/.linuxbrew/bin/thisroot.sh && \
    sudo git submodule update --init --recursive && \
    mkdir build_doc && \
    mkdir install_doc && \
    cd $GUNDIR/build_doc && \
    # for some reason yaml-cpp in not found by cmake, so put the paths manually
    cmake -DUSE_STATIC_LINKS=1 \
        -DCMAKE_INSTALL_PREFIX="$GUNDIR/install_doc" \
        -DYAMLCPP_INSTALL_DIR="/home/linuxbrew/.linuxbrew" \
        -DYAMLCPP_INCLUDE_DIR="/home/linuxbrew/.linuxbrew/include" \
        -DYAMLCPP_LIBRARY="/home/linuxbrew/.linuxbrew/lib/libyaml-cpp.so" \
        .. && \
    make -j6 && make install
