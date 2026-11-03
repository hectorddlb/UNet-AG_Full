#!/bin/bash
#SBATCH --job-name=v2_b1v4
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00
#SBATCH --output=slurm_logs/v2_b1v4_%j.out
#SBATCH --error=slurm_logs/v2_b1v4_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# B1-v4 — Rescue #1 on the closed B1/A2 family.
#
# Architecture identical to B1-v3 (residual skip + zero-init out + AG-Fd-F).
# ONLY difference: the loss adds a "lateral anchor" term L1(pred − x_DAS):
#
#   L = α·MSE + β_L1·L1(pred, t) + γ·(1−SSIM)
#     + anchor_w · L1(pred − x_DAS)        ← penalizes drift from the DAS_1PW floor
#     + gain_reg_w · log_gain²              ← optional
#
# Hypothesis: there is an anchor_w ∈ [0.3, 3.0] where the network improves
# gCNR without collapsing n_lat (verdict FWHM_lat<4.9 AND n_lat≥150/600). If NO
# value of the sweep meets it, the Pareto front is degenerate and the
# definitive closure of B1 is confirmed with mechanistic evidence.
#
# Smoke:
#   sbatch --export=EPOCHS=2,MAX_PHANTOMS_TRAIN=5,SMOKE_STEPS=6,ANCHOR_WEIGHT=1.0 slurm/train_b1_v4.sbatch
#
# Full (recommended sweep, 3 chained runs):
#   sbatch --export=ANCHOR_WEIGHT=1.0,OUTPUT_DIR=...results_v2/b1_v4_a1p0 slurm/train_b1_v4.sbatch
#   sbatch --export=ANCHOR_WEIGHT=0.3,OUTPUT_DIR=...results_v2/b1_v4_a0p3 slurm/train_b1_v4.sbatch
#   sbatch --export=ANCHOR_WEIGHT=3.0,OUTPUT_DIR=...results_v2/b1_v4_a3p0 slurm/train_b1_v4.sbatch

set -euo pipefail

mkdir -p slurm_logs

DATA_ROOT="${DATA_ROOT:-${SLURM_SUBMIT_DIR}/datasets/Field II/field_ii_data_multiangle}"
TARGET_ROOT="${TARGET_ROOT:-${SLURM_SUBMIT_DIR}/results_schiffner_targets_fieldii}"
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/results_v2/b1_v4}"
PHANTOM_TYPE="${PHANTOM_TYPE:-points}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-3}"
WD="${WD:-1e-4}"
NZ="${NZ:-612}"
NX="${NX:-388}"
BASE_CH="${BASE_CH:-32}"
EMBED_CH="${EMBED_CH:-16}"
MSE_WEIGHT="${MSE_WEIGHT:-0.7}"
L1_WEIGHT="${L1_WEIGHT:-0.2}"
SSIM_WEIGHT="${SSIM_WEIGHT:-0.1}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-1.0}"
GAIN_REG_WEIGHT="${GAIN_REG_WEIGHT:-0.0}"
INIT_GAIN="${INIT_GAIN:-1.0}"
SEED="${SEED:-42}"
MAX_PHANTOMS_TRAIN="${MAX_PHANTOMS_TRAIN:-}"
MAX_PHANTOMS_VAL="${MAX_PHANTOMS_VAL:-}"
SMOKE_STEPS="${SMOKE_STEPS:-}"
RESUME="${RESUME:-0}"
NO_AUG="${NO_AUG:-0}"
FREEZE_GAIN="${FREEZE_GAIN:-0}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) — $(date) ==="
echo "  data_root       : $DATA_ROOT"
echo "  target_root     : $TARGET_ROOT"
echo "  output_dir      : $OUTPUT_DIR"
echo "  phantom_type    : $PHANTOM_TYPE"
echo "  epochs          : $EPOCHS  lr : $LR  wd : $WD"
echo "  grid            : ${NZ}x${NX}  (NATIVE Schiffner grid)"
echo "  base_ch         : $BASE_CH  embed_ch : $EMBED_CH"
echo "  loss weights    : mse=$MSE_WEIGHT  l1=$L1_WEIGHT  ssim=$SSIM_WEIGHT"
echo "  RESCUE #1       : anchor_w=$ANCHOR_WEIGHT  gain_reg_w=$GAIN_REG_WEIGHT"
echo "  residual init   : gain=$INIT_GAIN  freeze_gain=$FREEZE_GAIN"
[ -n "$MAX_PHANTOMS_TRAIN" ] && echo "  max_train       : $MAX_PHANTOMS_TRAIN"
[ -n "$MAX_PHANTOMS_VAL"   ] && echo "  max_val         : $MAX_PHANTOMS_VAL"
[ -n "$SMOKE_STEPS"        ] && echo "  smoke_steps     : $SMOKE_STEPS"
{ nvidia-smi 2>&1 || true; } > /tmp/nvsmi_${SLURM_JOB_ID:-$$}.txt
head -15 /tmp/nvsmi_${SLURM_JOB_ID:-$$}.txt
rm -f /tmp/nvsmi_${SLURM_JOB_ID:-$$}.txt
echo "============================================"

EXTRA_ARGS=()
[ -n "$MAX_PHANTOMS_TRAIN" ] && EXTRA_ARGS+=(--max_phantoms_train "$MAX_PHANTOMS_TRAIN")
[ -n "$MAX_PHANTOMS_VAL"   ] && EXTRA_ARGS+=(--max_phantoms_val   "$MAX_PHANTOMS_VAL")
[ -n "$SMOKE_STEPS"        ] && EXTRA_ARGS+=(--smoke_steps        "$SMOKE_STEPS")
[ "$RESUME" = "1" ]          && EXTRA_ARGS+=(--resume)
[ "$NO_AUG" = "1" ]          && EXTRA_ARGS+=(--no_aug)
[ "$FREEZE_GAIN" = "1" ]     && EXTRA_ARGS+=(--freeze_gain)

python -u -m unet_ag.train.b1_v4_anchored \
    --data_root        "$DATA_ROOT" \
    --target_root      "$TARGET_ROOT" \
    --output_dir       "$OUTPUT_DIR" \
    --phantom_type     "$PHANTOM_TYPE" \
    --epochs           "$EPOCHS" --lr "$LR" --weight_decay "$WD" \
    --nz "$NZ" --nx "$NX" \
    --base_ch          "$BASE_CH" --embed_ch "$EMBED_CH" \
    --mse_weight       "$MSE_WEIGHT" \
    --l1_weight        "$L1_WEIGHT" \
    --ssim_weight      "$SSIM_WEIGHT" \
    --anchor_weight    "$ANCHOR_WEIGHT" \
    --gain_reg_weight  "$GAIN_REG_WEIGHT" \
    --init_gain        "$INIT_GAIN" \
    --seed             "$SEED" \
    --device cuda \
    "${EXTRA_ARGS[@]}"

echo "=== Done — $(date) ==="
