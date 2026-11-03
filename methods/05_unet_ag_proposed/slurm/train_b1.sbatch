#!/bin/bash
#SBATCH --job-name=v2_b1
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00
#SBATCH --output=slurm_logs/v2_b1_%j.out
#SBATCH --error=slurm_logs/v2_b1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# Proposal B1 — U-Net + Conditional Attention Gate by F-number.
# Defaults: 60 epochs, MSE+L1, AdamW, BatchNorm, supervised against
# Schiffner Fd-F 75 PW. See unet_ag/train/b1_attention_gate.py.
#
# Verify (smoke):
#   sbatch --export=EPOCHS=2,MAX_PHANTOMS_TRAIN=5,SMOKE_STEPS=6 slurm/train_b1.sbatch
# Full:
#   sbatch slurm/train_b1.sbatch

set -euo pipefail

mkdir -p slurm_logs

DATA_ROOT="${DATA_ROOT:-${SLURM_SUBMIT_DIR}/datasets/Field II/field_ii_data_multiangle}"
TARGET_ROOT="${TARGET_ROOT:-${SLURM_SUBMIT_DIR}/results_schiffner_targets_fieldii}"
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/results_v2/b1}"
PHANTOM_TYPE="${PHANTOM_TYPE:-points}"
EPOCHS="${EPOCHS:-60}"
LR="${LR:-1e-3}"
WD="${WD:-1e-4}"
NZ="${NZ:-612}"
NX="${NX:-388}"
BASE_CH="${BASE_CH:-32}"
EMBED_CH="${EMBED_CH:-16}"
L1_WEIGHT="${L1_WEIGHT:-0.1}"
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
echo "  data_root     : $DATA_ROOT"
echo "  target_root   : $TARGET_ROOT"
echo "  output_dir    : $OUTPUT_DIR"
echo "  phantom_type  : $PHANTOM_TYPE"
echo "  epochs        : $EPOCHS  lr : $LR  wd : $WD"
echo "  grid          : ${NZ}x${NX}  base_ch : $BASE_CH  embed_ch : $EMBED_CH"
[ -n "$MAX_PHANTOMS_TRAIN" ] && echo "  max_train     : $MAX_PHANTOMS_TRAIN"
[ -n "$MAX_PHANTOMS_VAL"   ] && echo "  max_val       : $MAX_PHANTOMS_VAL"
[ -n "$SMOKE_STEPS"        ] && echo "  smoke_steps   : $SMOKE_STEPS"
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

python -m unet_ag.train.b1_attention_gate \
    --data_root   "$DATA_ROOT" \
    --target_root "$TARGET_ROOT" \
    --output_dir  "$OUTPUT_DIR" \
    --phantom_type "$PHANTOM_TYPE" \
    --epochs "$EPOCHS" --lr "$LR" --weight_decay "$WD" \
    --nz "$NZ" --nx "$NX" \
    --base_ch "$BASE_CH" --embed_ch "$EMBED_CH" \
    --l1_weight "$L1_WEIGHT" \
    --seed "$SEED" \
    --device cuda \
    "${EXTRA_ARGS[@]}"

echo "=== Done — $(date) ==="
