#!/bin/bash
# Run one pipeline command on the compute server under a CPU nice level, with
# the project environment active and output logged under the project directory.
# Usage: run_stage.sh <echolabel sub-command> [extra arguments]
# Example: run_stage.sh audit-zarr --config configs/stage0_audit.yaml --paths configs/paths/server.yaml
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
log_dir="$ECHOLABEL_PROJECT_DIR/logs"
mkdir -p "$log_dir"
eval "$("${ECHOLABEL_CONDA:-conda}" shell.bash hook)"
conda activate "$ECHOLABEL_CONDA_ENV"
# A host with many cores shared between users: small numerical jobs crawl when every
# library opens one thread per core, so the thread pools are capped.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
cd "$ECHOLABEL_REPO_DIR"
cmd="$1"; shift
log="$log_dir/${cmd}_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "logging to $log"
nice -n 10 echolabel "$cmd" "$@" 2>&1 | tee "$log"
exit "${PIPESTATUS[0]}"
