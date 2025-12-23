#!/bin/bash
# Train every experiment, fold and seed of a model configuration on one GPU, one run
# after another, then evaluate them.
# Usage: run_stage8.sh [MIN_FREE_MIB] [--config file] [--experiments a,b] [--folds f1,f2] [--seeds 0,1,2]
# The configuration defaults to configs/example_acoustic_model.yaml, and experiments, folds
# and seeds to the lists in it. Finished runs are skipped by their manifests, so the script
# can be restarted.
set -uo pipefail
min_free="${1:-20000}"
shift 1 || shift $#
config=configs/example_acoustic_model.yaml
experiments=""; folds=""; seeds=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config) config="$2"; shift 2 ;;
    --experiments) experiments="$2"; shift 2 ;;
    --folds) folds="$2"; shift 2 ;;
    --seeds) seeds="$2"; shift 2 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$here/common.sh"
cd "$ECHOLABEL_REPO_DIR"
eval "$("${ECHOLABEL_CONDA:-conda}" shell.bash hook)"
conda activate "$ECHOLABEL_CONDA_ENV"
read_list() { python - "$config" "$1" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
key = sys.argv[2]
values = list(cfg["experiments"]) if key == "experiments" else cfg[key]
print(",".join(str(v) for v in values))
EOF
}
[ -n "$experiments" ] || experiments=$(read_list experiments)
[ -n "$folds" ] || folds=$(read_list folds)
[ -n "$seeds" ] || seeds=$(read_list seeds)
gpu=$(bash "$here/select_gpu.sh" "$min_free") || { echo "no GPU with $min_free MiB free"; exit 1; }
export CUDA_VISIBLE_DEVICES="$gpu"
echo "using GPU $gpu; experiments $experiments; folds $folds; seeds $seeds"
IFS=',' read -r -a exp_list <<< "$experiments"
IFS=',' read -r -a fold_list <<< "$folds"
IFS=',' read -r -a seed_list <<< "$seeds"
for e in "${exp_list[@]}"; do
  for f in "${fold_list[@]}"; do
    for s in "${seed_list[@]}"; do
      bash "$here/run_stage.sh" train-acoustic --config "$config" --paths "$ECHOLABEL_PATHS" --experiment "$e" --fold "$f" --seed "$s" || echo "FAILED $e $f $s"
    done
  done
done
bash "$here/run_stage.sh" evaluate-acoustic --config "$config" --paths "$ECHOLABEL_PATHS" --force
echo "all runs finished"
