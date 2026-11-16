#!/usr/bin/env bash

# Image identity and target platform.
CI_BASE_IMAGE=nadrino/gundam-ci-base
CI_BASE_VERSION=v1.0.0
CI_BASE_PLATFORM=linux/amd64

# Available ROOT/Ubuntu combinations: https://hub.docker.com/r/rootproject/root/tags
# Choose an existing tag matching <ROOT_VERSION>-ubuntu<UBUNTU_VERSION>.
CI_BASE_UBUNTU_VERSION=25.10
CI_BASE_ROOT_VERSION=6.38.00

# Keep C++ versions aligned with cmake/dependencies.cmake.
CI_BASE_JSON_VERSION=3.11.3
CI_BASE_YAML_CPP_VERSION=0.9.0
CI_BASE_GTEST_VERSION=1.16.0
CI_BASE_PYBIND11_VERSION=2.13.6
CI_BASE_BUILD_JOBS=3
