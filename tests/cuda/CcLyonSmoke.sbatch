#!/bin/bash
#SBATCH --job-name=gundam-cuda-smoke
#SBATCH --partition=gpu_v100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=gundam-cuda-smoke-%j.log

set -euo pipefail

# Submit from the repository root after compiling on an interactive GPU node:
# sbatch tests/cuda/CcLyonSmoke.sbatch [absolute-or-relative-build-directory]
# For an sm_90 build: sbatch -p gpu_h100 tests/cuda/CcLyonSmoke.sbatch ...
# Load the same toolkit environment here if it is not installed system-wide.
# The smoke test embeds its CUDA runtime path at link time; the toolkit must
# remain accessible at that path from the batch node.
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
buildDir="${1:-build-cuda-smoke}"
nvidia-smi
# Do not reuse an interactive session's CPU mask on a different batch node.
# This smoke test does not need CPU affinity.
srun --cpu-bind=none "${buildDir}/gundamCudaSmokeTest"
