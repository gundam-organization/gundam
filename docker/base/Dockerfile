# Dependencies only: no GUNDAM sources or submodules are included.
ARG ROOT_VERSION=6.34.00
ARG UBUNTU_VERSION=24.04
ARG ROOT_IMAGE=rootproject/root:${ROOT_VERSION}-ubuntu${UBUNTU_VERSION}
FROM ${ROOT_IMAGE} AS base

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake git libvdt-dev \
      python3-dev python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Keep these versions aligned with cmake/dependencies.cmake.
ARG JSON_VERSION=3.11.3
ARG YAML_CPP_VERSION=0.9.0
ARG GTEST_VERSION=1.16.0
ARG BUILD_JOBS=3

RUN git clone --depth 1 --branch "v${JSON_VERSION}" https://github.com/nlohmann/json.git /tmp/json && \
    cmake -S /tmp/json -B /tmp/json-build \
      -DCMAKE_INSTALL_PREFIX=/usr/local -DJSON_BuildTests=OFF && \
    cmake --install /tmp/json-build && \
    rm -rf /tmp/json /tmp/json-build

RUN git clone --depth 1 --branch "yaml-cpp-${YAML_CPP_VERSION}" https://github.com/jbeder/yaml-cpp.git /tmp/yaml-cpp && \
    cmake -S /tmp/yaml-cpp -B /tmp/yaml-build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF && \
    cmake --build /tmp/yaml-build --parallel "${BUILD_JOBS}" && \
    cmake --install /tmp/yaml-build && \
    rm -rf /tmp/yaml-cpp /tmp/yaml-build

RUN git clone --depth 1 --branch "v${GTEST_VERSION}" https://github.com/google/googletest.git /tmp/googletest && \
    cmake -S /tmp/googletest -B /tmp/gtest-build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    cmake --build /tmp/gtest-build --parallel "${BUILD_JOBS}" && \
    cmake --install /tmp/gtest-build && \
    rm -rf /tmp/googletest /tmp/gtest-build && ldconfig

ENV VIRTUAL_ENV=/opt/gundam-ci/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ARG PYBIND11_VERSION=2.13.6
COPY tests/python/requirements.txt /opt/gundam-ci/requirements.txt
RUN python3 -m venv "${VIRTUAL_ENV}" && \
    python -m pip install --no-cache-dir "pybind11==${PYBIND11_VERSION}" \
      -r /opt/gundam-ci/requirements.txt && \
    python -m pip check && \
    python -m pip freeze > /opt/gundam-ci/python-packages.txt

# Verify that the test packages can be imported without network access.
RUN --network=none python -c 'import pybind11, uproot, numpy, matplotlib'

WORKDIR /home/work
