#!/usr/bin/env bash
# Run inside a GPU allocation, with the CUDA-enabled GUNDAM Python module selected.
set -euo pipefail
if [[ $# -lt 1 || ${1:-} == --help ]]; then
  echo "Usage: $0 WORK_DIR [profiling options...]"
  echo 'Example: ./231CudaFormulaCompositeBenchmark.sh /path/to/upgrade-cc-2026 --calls 100 --trials 3'
  echo 'Requires Python with the CUDA-enabled GUNDAM module, and an allocated GPU.'
  echo 'Default overrides include onlyEb.yaml. Pass --override ... to replace the list.'
  exit 0
fi
benchmark_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
benchmark_work_dir=$1
shift
exec "${PYTHON:-python3}" "$benchmark_dir/229CudaBackendProfiling.py" \
  --work-dir "$benchmark_work_dir" --config ndUpCcInc.yaml \
  --override override/datasetHighlandv5.27.1_CCLyon.yaml override/ebMigrationDial.yaml \
    override/binResponse.yaml override/backendCpuTest.yaml override/onlyEb.yaml \
  --cases CPU CUDA-cachedResponses CUDA-recomputeResponses \
    CUDA-cachedResponses-interleaved CUDA-recomputeResponses-interleaved \
  --execution-mode isolated --event-variables Pmu --require-rebinning \
  --threads 4 --calls 100 --global-calls 40 --trials 3 --warmup 8 "$@"
