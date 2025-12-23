#!/bin/bash
# Run the transect pipeline for one detector run on the compute server: inference
# on the left camera of every 2018 transect, the frame review sets, the camera
# series and the alignment. Usage:
#   run_stage3.sh <experiment> <seed> [MIN_FREE_MIB] [--folders a,b] [--transects t1,t2]
# Inference selects the GPU with the most free memory; the other steps run on CPU.
set -uo pipefail
experiment="$1"; seed="$2"; min_free="${3:-20000}"
shift 3 || shift $#
folders=""; transects="20180505T0637Z,20180509T0018Z,20180509T0805Z,20180509T1938Z"
while [ $# -gt 0 ]; do
  case "$1" in
    --folders) folders="$2"; shift 2 ;;
    --transects) transects="$2"; shift 2 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$here/common.sh"
cd "$ECHOLABEL_REPO_DIR"
gpu=$(bash "$here/select_gpu.sh" "$min_free") || { echo "no GPU with $min_free MiB free"; exit 1; }
export CUDA_VISIBLE_DEVICES="$gpu"
export TORCH_HOME="$ECHOLABEL_SCRATCH_DIR/torch_home"
echo "using GPU $gpu for $experiment seed $seed"
common=(--config configs/stage2_detector.yaml --paths "$ECHOLABEL_PATHS" --experiment "$experiment" --seed "$seed")
IFS=',' read -r -a list <<< "$transects"
for t in "${list[@]}"; do
  if [ -n "$folders" ]; then
    bash "$here/run_stage.sh" infer-transect "${common[@]}" --transect "$t" --folders "$folders" || exit 1
  else
    bash "$here/run_stage.sh" infer-transect "${common[@]}" --transect "$t" || exit 1
  fi
done
for t in "${list[@]}"; do
  bash "$here/run_stage.sh" review-detections "${common[@]}" --transect "$t" || exit 1
  bash "$here/run_stage.sh" build-camera-series "${common[@]}" --transect "$t" || exit 1
  bash "$here/run_stage.sh" align-transect --config configs/stage6_alignment.yaml --paths "$ECHOLABEL_PATHS" --experiment "$experiment" --seed "$seed" --transect "$t" || exit 1
done
echo "stage 3, 4 and 6 finished for $experiment seed $seed"
