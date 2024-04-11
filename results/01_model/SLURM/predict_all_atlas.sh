#!/bin/bash

source "$HOME/.bashrc"
set -euo pipefail

rsync \
  --archive \
  tmp/dataset \
  "$TMPDIR"

export A2ZE_PREDICT_OUTPUT_PATH="tmp/predictions.${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tsv"
export A2ZE_PREDICT_RUN_INDEX=$SLURM_ARRAY_TASK_ID
pixi run predict-atlas

# clean up
rm --recursive --force "${TMPDIR}/dataset"
