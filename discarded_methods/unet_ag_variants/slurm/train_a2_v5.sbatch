#!/bin/bash
#SBATCH --job-name=v2_a2v5
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00
#SBATCH --output=slurm_logs/v2_a2v5_%j.out
#SBATCH --error=slurm_logs/v2_a2v5_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# A2-v5 — A2-v4 (adjoint input + 2-D Fourier-weighted loss) at NATIVE GRID.
# Root cause for A2-v4 lateral plateau at ~7.77mm: training grid (256,128)
# bilinearly downsamples the (612,388) Schiffner target, smearing wires from
# native FWHM 1-3mm to 7-14mm. A2-v5 keeps everything identical to A2-v4 but
# trains at the target's native (612,388) grid. EPOCHS reduced to 25 to fit
# the 18h wall (≈7.25× pixels per epoch vs A2-v4).
#
# Smoke:
#   sbatch --export=EPOCHS=2,MAX_PHANTOMS_TRAIN=5,SMOKE_STEPS=6 slurm/train_a2_v5.sbatch
# Full:
#   sbatch slurm/train_a2_v5.sbatch

set -euo pipefail

mkdir -p slurm_logs

DATA_ROOT="${DATA_ROOT:-${SLURM_SUBMIT_DIR}/datasets/Field II/field_ii_data_multiangle}"
TARGET_ROOT="${TARGET_ROOT:-${SLURM_SUBMIT_DIR}/results_schiffner_targets_fieldii}"
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/results_v2/a2_v5}"
PHANTOM_TYPE="${PHANTOM_TYPE:-points}"
EPOCHS="${EPOCHS:-25}"
LR="${LR:-1e-3}"
WD="${WD:-1e-4}"
NZ="${NZ:-612}"
NX="${NX:-388}"
BASE_CH="${BASE_CH:-32}"
EMBED_CH="${EMBED_CH:-16}"
MSE_WEIGHT="${MSE_WEIGHT:-0.7}"
L1_WEIGHT="${L1_WEIGHT:-0.2}"
FOURIER_WEIGHT="${FOURIER_WEIGHT:-0.15}"
FOURIER_MODE="${FOURIER_MODE:-inv_F_squared}"
LATERAL_MODE="${LATERAL_MODE:-highpass_linear}"
LATERAL_FLOOR="${LATERAL_FLOOR:-0.1}"
SEED="${SEED:-42}"
MAX_PHANTOMS_TRAIN="${MAX_PHANTOMS_TRAIN:-}"
MAX_PHANTOMS_VAL="${MAX_PHANTOMS_VAL:-}"
SMOKE_STEPS="${SMOKE_STEPS:-}"
RESUME="${RESUME:-0}"
NO_AUG="${NO_AUG:-0}"
NO_FC_OFFSET="${NO_FC_OFFSET:-0}"

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
echo "  loss weights    : mse=$MSE_WEIGHT  l1=$L1_WEIGHT  fourier=$FOURIER_WEIGHT"
echo "  axial mode      : $FOURIER_MODE  no_fc_offset=$NO_FC_OFFSET"
echo "  lateral         : mode=$LATERAL_MODE  floor=$LATERAL_FLOOR"
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
[ "$NO_FC_OFFSET" = "1" ]    && EXTRA_ARGS+=(--no_fc_offset)

python -u -m unet_ag.train.a2_v4_adjoint_fourier \
    --data_root      "$DATA_ROOT" \
    --target_root    "$TARGET_ROOT" \
    --output_dir     "$OUTPUT_DIR" \
    --phantom_type   "$PHANTOM_TYPE" \
    --epochs         "$EPOCHS" --lr "$LR" --weight_decay "$WD" \
    --nz "$NZ" --nx "$NX" \
    --base_ch        "$BASE_CH" --embed_ch "$EMBED_CH" \
    --mse_weight     "$MSE_WEIGHT" \
    --l1_weight      "$L1_WEIGHT" \
    --fourier_weight "$FOURIER_WEIGHT" \
    --fourier_mode   "$FOURIER_MODE" \
    --lateral_mode   "$LATERAL_MODE" \
    --lateral_floor  "$LATERAL_FLOOR" \
    --seed           "$SEED" \
    --device cuda \
    "${EXTRA_ARGS[@]}"

echo "=== Done — $(date) ==="
