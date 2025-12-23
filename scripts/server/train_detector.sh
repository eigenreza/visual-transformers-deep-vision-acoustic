#!/bin/bash
# Train one detector experiment and seed on the GPU with the most free memory,
# then run inference and evaluation on validation, test and the 2024 hold-out.
# Usage: train_detector.sh <experiment> <seed> [MIN_FREE_MIB] [extra train-detector arguments]
# Example: train_detector.sh dfine_l_640 0 30000
set -uo pipefail
exp="$1"; seed="$2"; min_free="${3:-30000}"; shift 3 || shift $#
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$here/common.sh"
run="$here/run_stage.sh"
idx=$(bash "$here/select_gpu.sh" "$min_free") || { echo "no GPU with $min_free MiB free"; exit 2; }
export CUDA_VISIBLE_DEVICES="$idx"
export TORCH_HOME="$ECHOLABEL_SCRATCH_DIR/pretrained/torch"
args="--config configs/stage2_detector.yaml --paths $ECHOLABEL_PATHS --experiment $exp --seed $seed"
bash "$run" train-detector $args "$@" || exit $?
for split in val test test_2024; do
  bash "$run" predict-detector $args --split "$split" || exit $?
done
for split in val test test_2024; do
  bash "$run" evaluate-detector $args --split "$split" || exit $?
done
bash "$run" calibrate-detector $args || exit $?
