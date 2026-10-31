#!/usr/bin/env bash
#SBATCH --job-name=gundam-weights-v100
#SBATCH --partition=gpu_v100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=gundam-weights-v100-%j.log
# Submit in the same ROOT/Python/CUDA environment used to build GUNDAM.
set -euo pipefail
: "${GUNDAM_REPO:?Export GUNDAM_REPO before sbatch}"
exec "${PYTHON:-python3}" "$GUNDAM_REPO/tests/benchmarks/236CudaWeightBenchmark.py" "$@"
