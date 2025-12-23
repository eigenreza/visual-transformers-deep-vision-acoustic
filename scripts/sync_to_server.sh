#!/bin/bash
# Copy the repository (tracked files and untracked files that are not ignored)
# to the compute server. Run from the development machine.
# Usage: bash scripts/sync_to_server.sh
# The login and the repository directory on the server are read from scripts/server.env
# or from the environment (ECHOLABEL_REMOTE, ECHOLABEL_REPO_DIR).
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -f "$repo_dir/scripts/server.env" ]; then set -a; . "$repo_dir/scripts/server.env"; set +a; fi
remote="${ECHOLABEL_REMOTE:?set ECHOLABEL_REMOTE (see scripts/server.env.example)}"
remote_dir="${ECHOLABEL_REPO_DIR:?set ECHOLABEL_REPO_DIR (see scripts/server.env.example)}"
cd "$repo_dir"
git ls-files -co --exclude-standard -z | tar --null -T - -cf - | \
  ssh "$remote" "mkdir -p '$remote_dir' && tar -xf - -C '$remote_dir'"
echo "synced to $remote:$remote_dir"
