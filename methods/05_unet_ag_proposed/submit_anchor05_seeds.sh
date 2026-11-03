#!/bin/bash
# 3-seed defense statistics on B1-v5 anchor05 (β=0.5) — winner of B1 campaign.
#
# anchor05 with SEED=42 already trained → results_v2/b1_v5_anchor05/
# This adds SEED=1 → results_v2/b1_v5_anchor05_s1/
#         SEED=2 → results_v2/b1_v5_anchor05_s2/
#
# Each chain: train + eval_fii + eval_picmus.
# afterany throughout.

set -euo pipefail
cd "$(dirname "$0")/../.."   # → repo root
REPO=$(pwd)

submit_seed() {
    local SEED="$1"
    local NAME="anchor05_s${SEED}"
    local OUT_DIR="${REPO}/results_v2/b1_v5_${NAME}"
    local DEP="${2:-}"

    local DEP_FLAG=""
    if [[ -n "$DEP" ]]; then
        DEP_FLAG="--dependency=afterany:$DEP"
    fi

    local T
    T=$(sbatch $DEP_FLAG \
        --export=ALL,OUTPUT_DIR=$OUT_DIR,SEED=$SEED,ANCHOR_WEIGHT=0.5 \
        --job-name=v2_b1v5_${NAME} \
        unet_ag/slurm/train_b1_v5.sbatch | awk '{print $4}')

    local EFII
    EFII=$(sbatch --dependency=afterany:$T \
        --export=ALL,CKPT=${OUT_DIR}/checkpoints/best.pt,OUT=${OUT_DIR}/eval_fieldii.csv \
        --job-name=v2_efii_${NAME} \
        unet_ag/slurm/eval_b1_v4_fieldii.sbatch | awk '{print $4}')

    local EPICMUS
    EPICMUS=$(sbatch --dependency=afterany:$EFII \
        --export=ALL,CKPT=${OUT_DIR}/checkpoints/best.pt,OUT=${OUT_DIR}/eval_picmus.csv \
        --job-name=v2_epicmus_${NAME} \
        unet_ag/slurm/eval_picmus.sbatch | awk '{print $4}')

    echo "${NAME}: train=$T  eval_fii=$EFII  eval_picmus=$EPICMUS"
    LAST="$EPICMUS"
}

LAST=""
submit_seed 1
submit_seed 2 "$LAST"

echo
echo "Chain end: $LAST"
echo "Total queued: $(squeue -u $USER -h | wc -l) jobs"
squeue -u $USER
