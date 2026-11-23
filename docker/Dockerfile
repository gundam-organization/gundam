# Published dependency image; override with --build-arg CI_BASE_IMAGE=... .
ARG CI_BASE_IMAGE=nadrino/gundam-ci-base:v1.0.0
FROM ${CI_BASE_IMAGE} AS base

SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

ENV WORK_DIR=/home/work
ENV REPO_DIR=${WORK_DIR}/repo
ENV BUILD_DIR=${WORK_DIR}/build
ENV INSTALL_DIR=${WORK_DIR}/install

# Submodules must be populated in the checkout before building this image.
COPY ./src ${REPO_DIR}/src
COPY ./submodules ${REPO_DIR}/submodules
COPY ./cmake ${REPO_DIR}/cmake
COPY ./CMakeLists.txt ${REPO_DIR}/CMakeLists.txt
COPY ./.git ${REPO_DIR}/.git
COPY ./tests ${REPO_DIR}/tests

# All dependencies and the Python environment are supplied by CI_BASE_IMAGE.
# Disable network access for configuration, compilation and tests.
WORKDIR ${BUILD_DIR}
RUN --network=none cmake \
      -D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
      -D WITH_PYTHON_INTERFACE=ON \
      -D WITH_PYTHON_TEST_VENV=OFF \
      -D GUNDAM_PYTHON_VERSIONS=${VIRTUAL_ENV}/bin/python \
      -D FETCHCONTENT_FULLY_DISCONNECTED=ON \
      -D pybind11_DIR="$(python -m pybind11 --cmakedir)" \
      ${REPO_DIR}
RUN --network=none cmake --build . --parallel 3 --target install

RUN --network=none source ${INSTALL_DIR}/setup.sh && \
    ctest --output-on-failure

ENV PATH=${INSTALL_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${INSTALL_DIR}/lib:${PYTHONPATH}

WORKDIR /home
