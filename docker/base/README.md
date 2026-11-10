# CI dependency image

`Dockerfile` builds a reusable environment without building GUNDAM.
It contains ROOT, the compiler and build tools, JSON 3.11.3, yaml-cpp 0.9.0,
GoogleTest 1.16.0, and a Python virtual environment with pybind11 2.13.6 and
`tests/python/requirements.txt`. C++ dependencies are installed in `/usr/local`;
Python is available on `PATH` through `/opt/gundam-ci/venv`.

## Build and publish

Edit `docker/base/versions.sh` to choose the image tag, ROOT/Ubuntu versions,
dependency versions, target platform and build parallelism. The script loads
these variables and explicitly passes them as Docker build arguments; they do
not need to be exported. Dockerfile defaults remain available for direct builds.

Run these commands from the repository root. The build script always uses the
repository root as context so Docker can copy `tests/python/requirements.txt`,
even when launched from another directory.
Python packages are installed and their imports checked inside the Dockerfile;
no external installation command is needed.

```bash
bash docker/base/build.sh
source docker/base/versions.sh
docker login
docker push "$CI_BASE_IMAGE:$CI_BASE_VERSION"
```

The tag is assigned locally by `-t` at build time; `docker push` publishes that
tag. To publish the same image under another tag without rebuilding:

```bash
docker tag "$CI_BASE_IMAGE:$CI_BASE_VERSION" "$CI_BASE_IMAGE:root6.34.00-ubuntu24.04-v1.0.0"
docker push "$CI_BASE_IMAGE:root6.34.00-ubuntu24.04-v1.0.0"
```

For the next base revision, change `CI_BASE_VERSION` in `versions.sh`, rebuild
and push. Keep published
version tags unchanged so existing CIs keep using the same environment.

`linux/amd64` targets the existing Ubuntu GitHub runner, including when building
on an Apple Silicon Mac. ROOT's selected tag must support the target platform.
Additional Docker build options can be passed to the script, for example
`bash docker/base/build.sh --no-cache` or
`bash docker/base/build.sh --build-arg ROOT_IMAGE=...`.

Use a new tag for every dependency update. Python requirements currently have
no version pins: rebuilding the base may resolve newer packages. Each published
image freezes the installed environment; its exact Python versions are recorded
in `/opt/gundam-ci/python-packages.txt`. For reproducible CI consumption, record
the pushed image digest and use `image@sha256:...` in the future CI configuration.

## Build GUNDAM from the published image

The [`docker/Dockerfile`](../Dockerfile) uses `nadrino/gundam-ci-base:v1.0.0` by default.
It copies the local sources and submodules, then configures, compiles, installs
and tests GUNDAM with network access disabled. No apt, pip or Git downloads
are performed during these steps.

```bash
git submodule update --init --recursive
docker build --platform linux/amd64 --network=none -f docker/Dockerfile -t nadrino/gundam:local .
```

To select a newer published dependency image:

```bash
docker build --platform linux/amd64 --network=none -f docker/Dockerfile \
  --build-arg CI_BASE_IMAGE=nadrino/gundam-ci-base:v1.1.0 \
  -t nadrino/gundam:local .
```

Update the default in `docker/Dockerfile` and `CI_BASE_IMAGE` in the release
workflow when adopting a new base version in CI. The image can also be selected
by digest. `docker/base/versions.sh` controls building the dependency image;
changing it does not automatically change the base consumed by GUNDAM.

The repository checkout (including submodules) and image pull still require
network access. Dependency downloads only happen when building the base image.
Rebuild it when ROOT, C++ dependency versions, or Python requirements change.
