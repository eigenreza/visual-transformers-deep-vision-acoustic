#!/bin/bash
# Create the conda environment of the pipeline on the compute server from the pinned lock file.
# Re-runnable: environment creation is skipped when the interpreter already exists,
# and pip only installs what is missing or differs from the lock.
# Usage: bash setup_env.sh
# ECHOLABEL_CONDA_ENV must be the prefix of the environment (see scripts/server.env.example).
set -uo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$script_dir/common.sh"
env_prefix="$ECHOLABEL_CONDA_ENV"
lock="$ECHOLABEL_REPO_DIR/requirements/server-lock.txt"
mkdir -p "$ECHOLABEL_PROJECT_DIR/logs"
log="$ECHOLABEL_PROJECT_DIR/logs/env_setup.log"
exec > >(tee -a "$log") 2>&1
echo "=== setup_env start $(date -u +%FT%TZ) ==="
[ -f "$lock" ] || { echo "lock file not found: $lock"; exit 1; }
eval "$("${ECHOLABEL_CONDA:-conda}" shell.bash hook)"
if [ ! -x "$env_prefix/bin/python" ]; then
  nice -n 10 conda create -y -p "$env_prefix" python=3.11.16 pip || { echo "conda create failed"; exit 1; }
fi
conda activate "$env_prefix"
which python
python --version
nice -n 10 python -m pip install --no-cache-dir -r "$lock" || { echo "pip install failed"; exit 1; }
python -m pip freeze > "$ECHOLABEL_PROJECT_DIR/logs/pip_freeze.txt"
python -c "import torch, torchvision, xarray, zarr, dask, numpy, pandas, pyarrow, PIL, lxml, scipy, sklearn; print('imports ok; torch', torch.__version__, 'cuda available', torch.cuda.is_available())"
echo "=== setup_env done $(date -u +%FT%TZ) ==="
