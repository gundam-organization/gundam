# GUNDAM Docker image

[`Dockerfile`](Dockerfile) builds GUNDAM from the current checkout using
`nadrino/gundam-ci-base:v1.0.0`. The dependency image is maintained separately;
see [base/README.md](base/README.md) to build and publish it.

## Build

Run from the repository root, with Docker running:

```bash
git submodule update --init --recursive
docker build --platform linux/amd64 --network=none \
  -f docker/Dockerfile -t nadrino/gundam:local .
```

The final `.` must be the repository root: the Dockerfile copies the source,
tests, Git metadata and populated submodules from that context. Build from a
regular Git checkout, rather than a source archive without `.git`.

Configuration, compilation, installation and CTest all run during the build
with network access disabled. A failing test fails the image build. Dependencies
are already installed in the base image; fetching that image and checking out
the repository/submodules still require network access.

`linux/amd64` matches the GitHub Ubuntu runners and also selects that target
when building on an Apple Silicon Mac.

To use another published dependency image, add
`--build-arg CI_BASE_IMAGE=nadrino/gundam-ci-base:v1.1.0` (or an image digest).
When adopting a new default in CI, update both `docker/Dockerfile` and
`CI_BASE_IMAGE` in `.github/workflows/docker-publish-release.yml`.

## Run

Start an interactive shell:

```bash
docker run --rm -it --platform linux/amd64 nadrino/gundam:local bash
```

GUNDAM is installed in `/home/work/install`; its executables and Python module
are available through the image's environment. The build is in `/home/work/build`
and its source checkout in `/home/work/repo`.

To run a fit using files in the current directory and keep outputs on the host:

```bash
docker run --rm --platform linux/amd64 \
  -v "$PWD:/work" -w /work \
  nadrino/gundam:local gundamFitter -c config.yaml
```

Replace `config.yaml` with your configuration; input paths must be accessible
inside the container. On Linux, add `--user "$(id -u):$(id -g)"` if outputs should
belong to your host user.

To rerun the tests in an existing image without rebuilding:

```bash
docker run --rm --platform linux/amd64 --network none \
  -w /home/work/build nadrino/gundam:local \
  bash -c 'source /home/work/install/setup.sh && ctest --output-on-failure'
```

Docker Compose is unnecessary for this single-container workflow. The former
root-level Compose file used obsolete Linuxbrew paths and has been removed.
