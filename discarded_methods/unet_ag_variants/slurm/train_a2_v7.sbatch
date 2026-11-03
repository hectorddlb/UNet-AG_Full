#!/bin/bash
#SBATCH --job-name=v2_a2v7
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00
#SBATCH --output=slurm_logs/v2_a2v7_%j.out
#SBATCH --error=slurm_logs/v2_a2v7_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# A2-v7 — Rescue of the B1/A2 closure via an edge-sharpness loss.
#
# Same backbone as A2-v6 (B1UNetAttentionFNum, base_ch=48 ≈ 7M params), same
# input (adjoint env+log + f_map), same target (Schiffner_75PW env+log on the
# NATIVE 612×388 grid). ONLY difference: the loss adds an L1 on the residual
# of the lateral derivative:
#
#   L = α·MSE_w + β·L1_w + γ·L1(∂_x e) + δ·L1(∂_z e)
#
# where e = pred − target, MSE_w/L1_w are target-weighted (w = 1 + α_w·t).
#
# Hypothesis: γ ∈ [0.3, 3.0] rescues FWHM_lat without breaking gCNR. If NO γ
# of the sweep passes the verdict (FWHM_lat<4.9 AND n_lat≥150/600) then the
# edge-vs-body Pareto front is also degenerate and the definitive closure of
# the family is confirmed (with double mechanistic evidence: B1-v4 anchor
# and A2-v7 derivative).
#
# Smoke:
#   sbatch --export=EPOCHS=2,MAX_PHANTOMS_TRAIN=5,SMOKE_STEPS=6 slurm/train_a2_v7.sbatch
#
# Sweep (3 chained runs, ~27h total):
#   sbatch --export=LATERAL_WEIGHT=0.3,OUTPUT_DIR=...results_v2/a2_v7_l0p3 slurm/train_a2_v7.sbatch
#   sbatch --export=LATERAL_WEIGHT=1.0,OUTPUT_DIR=...results_v2/a2_v7_l1p0 slurm/train_a2_v7.sbatch
#   sbatch --export=LATERAL_WEIGHT=3.0,OUTPUT_DIR=...results_v2/a2_v7_l3p0 slurm/train_a2_v7.sbatch

set -euo pipefail

mkdir -p slurm_logs

DATA_ROOT="${DATA_ROOT:-${SLURM_SUBMIT_DIR}/datasets/Field II/field_ii_data_multiangle}"
TARGET_ROOT="${TARGET_ROOT:-${SLURM_SUBMIT_DIR}/results_schiffner_targets_fieldii}"
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/results_v2/a2_v7}"
PHANTOM_TYPE="${PHANTOM_TYPE:-points}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-3}"
WD="${WD:-1e-4}"
NZ="${NZ:-612}"
NX="${NX:-388}"
BASE_CH="${BASE_CH:-48}"
EMBED_CH="${EMBED_CH:-16}"
MSE_WEIGHT="${MSE_WEIGHT:-0.5}"
L1_WEIGHT="${L1_WEIGHT:-0.15}"
LATERAL_WEIGHT="${LATERAL_WEIGHT:-0.3}"
AXIAL_WEIGHT="${AXIAL_WEIGHT:-0.05}"
ALPHA="${ALPHA:-9.0}"
SEED="${SEED:-42}"
MAX_PHANTOMS_TRAIN="${MAX_PHANTOMS_TRAIN:-}"
MAX_PHANTOMS_VAL="${MAX_PHANTOMS_VAL:-}"
SMOKE_STEPS="${SMOKE_STEPS:-}"
RESUME="${RESUME:-0}"
NO_AUG="${NO_AUG:-0}"

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
echo "  body loss       : mse_w=$MSE_WEIGHT  l1_w=$L1_WEIGHT  alpha=$ALPHA"
echo "  RESCUE #A2-v7   : lateral_w=$LATERAL_WEIGHT  axial_w=$AXIAL_WEIGHT"
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

python -u -m unet_ag.train.a2_v7_lateral_deriv \
    --data_root        "$DATA_ROOT" \
    --target_root      "$TARGET_ROOT" \
    --output_dir       "$OUTPUT_DIR" \
    --phantom_type     "$PHANTOM_TYPE" \
    --epochs           "$EPOCHS" --lr "$LR" --weight_decay "$WD" \
    --nz "$NZ" --nx "$NX" \
    --base_ch          "$BASE_CH" --embed_ch "$EMBED_CH" \
    --mse_weight       "$MSE_WEIGHT" \
    --l1_weight        "$L1_WEIGHT" \
    --lateral_weight   "$LATERAL_WEIGHT" \
    --axial_weight     "$AXIAL_WEIGHT" \
    --alpha            "$ALPHA" \
    --seed             "$SEED" \
    --device cuda \
    "${EXTRA_ARGS[@]}"

echo "=== Done — $(date) ==="
