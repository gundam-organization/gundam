#!/usr/bin/env bash
#SBATCH --job-name=gundam-llh-v100
#SBATCH --partition=gpu_v100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=gundam-llh-v100-%j.log
# Submit after activating the ROOT/CUDA/Python environment used to build GUNDAM.
# GUNDAM_REPO must point to the checkout on the CC filesystem.
set -euo pipefail
: "${GUNDAM_REPO:?Export GUNDAM_REPO before sbatch}"
exec "${PYTHON:-python3}" "$GUNDAM_REPO/tests/benchmarks/235CudaV100LikelihoodBenchmark.py" "$@"
