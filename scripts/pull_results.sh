#!/bin/bash
# Copy small result tables (CSV, JSON) of a stage from the compute server into
# results/ in the repository. Large per-item tables (Parquet) stay on the server.
# Machine-specific roots in the copies are replaced by placeholders.
# Usage: bash scripts/pull_results.sh <run subdirectory> [<run subdirectory> ...]
# Example: bash scripts/pull_results.sh audit/annotations audit/transects audit/zarr
# The login and the directories of the server are read from scripts/server.env or from
# the environment (see scripts/server.env.example).
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -f "$repo_dir/scripts/server.env" ]; then set -a; . "$repo_dir/scripts/server.env"; set +a; fi
remote="${ECHOLABEL_REMOTE:?set ECHOLABEL_REMOTE (see scripts/server.env.example)}"
remote_runs="${ECHOLABEL_PROJECT_DIR:?set ECHOLABEL_PROJECT_DIR (see scripts/server.env.example)}/runs"
for sub in "$@"; do
  dest="$repo_dir/results/$sub"
  mkdir -p "$dest"
  ssh "$remote" "cd '$remote_runs/$sub' && tar -cf - --exclude='*.parquet' --exclude='run.log' --exclude='*_cache.json' --exclude='coco' --exclude='*.log' --exclude='framework' --exclude='*.npz' ." \
    | tar -xf - -C "$dest"
  python "$repo_dir/scripts/redact_paths.py" --paths "$repo_dir/configs/paths/server.example.yaml" "$dest"
  echo "pulled $sub -> results/$sub"
done
