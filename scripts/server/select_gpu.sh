#!/bin/bash
# Print the index of the GPU with the most free memory.
# Ties are broken by the lowest reported utilization.
# Usage: select_gpu.sh [MIN_FREE_MIB]
# If MIN_FREE_MIB is given and no GPU has that much free memory, exit with status 2.
# The full GPU table is written to standard error so that job logs record the
# state of the machine at launch.
set -euo pipefail
min_free="${1:-0}"
table=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
printf 'gpu state at launch (index, used MiB, total MiB, util %%):\n%s\n' "$table" >&2
best=$(printf '%s\n' "$table" | awk -F', ' '{printf "%d %d %d\n", $1, $3 - $2, $4}' | sort -k2,2nr -k3,3n | head -n 1)
idx=$(printf '%s' "$best" | cut -d' ' -f1)
free=$(printf '%s' "$best" | cut -d' ' -f2)
if [ "$free" -lt "$min_free" ]; then
  printf 'no GPU has %s MiB free (best is GPU %s with %s MiB)\n' "$min_free" "$idx" "$free" >&2
  exit 2
fi
printf 'selected GPU %s with %s MiB free\n' "$idx" "$free" >&2
printf '%s\n' "$idx"
