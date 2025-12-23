#!/bin/bash
# Settings shared by the launch scripts. The file scripts/server.env is read when it
# exists; otherwise the variables must be exported (see scripts/server.env.example).
common_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "$common_dir/../server.env" ]; then
  set -a
  . "$common_dir/../server.env"
  set +a
fi
: "${ECHOLABEL_REPO_DIR:=$(cd "$common_dir/../.." && pwd)}"
: "${ECHOLABEL_PATHS:=configs/paths/server.yaml}"
for name in ECHOLABEL_PROJECT_DIR ECHOLABEL_SCRATCH_DIR ECHOLABEL_CONDA_ENV; do
  [ -n "${!name:-}" ] || { echo "$name is not set; see scripts/server.env.example" >&2; exit 2; }
done
export ECHOLABEL_REPO_DIR ECHOLABEL_PATHS ECHOLABEL_PROJECT_DIR ECHOLABEL_SCRATCH_DIR ECHOLABEL_CONDA_ENV
export ECHOLABEL_SOURCE_DATA ECHOLABEL_ANNOTATED_DATA ECHOLABEL_ANNOTATED_DATA_SOURCE
