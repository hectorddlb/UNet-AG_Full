#!/bin/bash
#SBATCH --job-name=v2_ft_picmus
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/v2_ft_picmus_%j.out
#SBATCH --error=slurm_logs/v2_ft_picmus_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# B1-v4 fine-tuning on PICMUS — OOD penalty diagnosis.
#
# 3 modes available via TRAIN_MODE:
#   gain_only      → log_gain only (1 param). Tests whether OOD is pure scaling.
#   decoder_only   → log_gain + UNet decoder (~½ of the model).
#   full           → the whole model (3.5M params).
#
# Train: 2 PICMUS sim phantoms (resolution + contrast).
# Val:   2 PICMUS exp phantoms (resolution + contrast).
# Target: PICMUS reconstructed_image npw=75 (env+log compressed).

set -euo pipefail

mkdir -p slurm_logs

CKPT="${CKPT:-${SLURM_SUBMIT_DIR}/results_v2/b1_v4_a0p3/checkpoints/best.pt}"
PICMUS_ROOT="${PICMUS_ROOT:-${SLURM_SUBMIT_DIR}/datasets/PICMUS}"
TRAIN_MODE="${TRAIN_MODE:-gain_only}"
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/results_v2/b1_v4_ft_picmus_${TRAIN_MODE}}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-5}"
WD="${WD:-1e-3}"
ANCHOR_W="${ANCHOR_W:-0.5}"
BASE_CH="${BASE_CH:-32}"
SEED="${SEED:-42}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) — $(date) ==="
echo "  ckpt          : $CKPT"
echo "  picmus_root   : $PICMUS_ROOT"
echo "  train_mode    : $TRAIN_MODE"
echo "  output_dir    : $OUTPUT_DIR"
echo "  epochs        : $EPOCHS  lr: $LR  wd: $WD"
echo "  anchor_weight : $ANCHOR_W"
{ nvidia-smi 2>&1 || true; } > /tmp/nvsmi_${SLURM_JOB_ID:-$$}.txt
head -15 /tmp/nvsmi_${SLURM_JOB_ID:-$$}.txt
rm -f /tmp/nvsmi_${SLURM_JOB_ID:-$$}.txt
echo "============================================"

python -u -m unet_ag.train.b1_v4_finetune_picmus \
    --ckpt          "$CKPT" \
    --picmus_root   "$PICMUS_ROOT" \
    --output_dir    "$OUTPUT_DIR" \
    --train_mode    "$TRAIN_MODE" \
    --epochs        "$EPOCHS" \
    --lr            "$LR" \
    --weight_decay  "$WD" \
    --anchor_weight "$ANCHOR_W" \
    --base_ch       "$BASE_CH" \
    --seed          "$SEED" \
    --device        cuda

echo "=== Done — $(date) ==="
