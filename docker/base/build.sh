#!/usr/bin/env bash
set -euo pipefail

scriptDir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repoDir="$(cd -- "${scriptDir}/../.." && pwd)"
source "${scriptDir}/versions.sh"

imageTag="${CI_BASE_IMAGE}:${CI_BASE_VERSION}"
docker buildx build \
  --platform "${CI_BASE_PLATFORM}" --load \
  --file "${scriptDir}/Dockerfile" \
  --tag "${imageTag}" \
  --build-arg "ROOT_VERSION=${CI_BASE_ROOT_VERSION}" \
  --build-arg "UBUNTU_VERSION=${CI_BASE_UBUNTU_VERSION}" \
  --build-arg "JSON_VERSION=${CI_BASE_JSON_VERSION}" \
  --build-arg "YAML_CPP_VERSION=${CI_BASE_YAML_CPP_VERSION}" \
  --build-arg "GTEST_VERSION=${CI_BASE_GTEST_VERSION}" \
  --build-arg "PYBIND11_VERSION=${CI_BASE_PYBIND11_VERSION}" \
  --build-arg "BUILD_JOBS=${CI_BASE_BUILD_JOBS}" \
  "$@" "${repoDir}"

printf '\nImage built: %s\nTo publish: docker push %s\n' "${imageTag}" "${imageTag}"
