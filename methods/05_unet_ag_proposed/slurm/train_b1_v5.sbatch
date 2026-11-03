#!/bin/bash
#SBATCH --job-name=v2_b1v5
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00
#SBATCH --output=slurm_logs/v2_b1v5_%j.out
#SBATCH --error=slurm_logs/v2_b1v5_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# B1-v5 — B1-v4(0.3) + pulse-shape augmentation runtime.
#
# Motivation: the 2026-05-29 spectral measurement revealed a Field II ↔ PICMUS gap:
#   Field II training:        fc_peak=4.97 MHz   BW(-6dB)=1.82 MHz
#   PICMUS exp resolution:    fc_peak=5.04 MHz   BW(-6dB)=1.04 MHz
# Diagnosis (B) job 192069: the (fc, BW) grid search confirmed that a bandpass
# pre-applied to the PICMUS RF reduces val_total by > 10% on sim/res → a real
# and recoverable spectral gap. B1-v5 applies it DURING FULL TRAINING instead
# of post-hoc to mitigate the resolution OOD penalty.
#
# Defaults validados:
#   pulse_bw_range_hz = [1.0, 2.5] MHz  — covers PICMUS exp 1.04 → Field II 1.82 + margin
#   pulse_fc_shift_hz = ±300 kHz        — covers the measured range [4.64, 5.04] MHz
#   pulse_apply_prob  = 0.7              — high but not 1.0 (keeps exposure to native Field II)
#   anchor_weight     = 0.3              — winning value de B1-v4
#
# Smoke:
#   sbatch --export=EPOCHS=2,MAX_PHANTOMS_TRAIN=5,SMOKE_STEPS=6 slurm/train_b1_v5.sbatch
#
# Full:
#   sbatch slurm/train_b1_v5.sbatch                                   # 20 ep, ~9 h
#   sbatch --export=OUTPUT_DIR=...results_v2/b1_v5_pulse slurm/train_b1_v5.sbatch

set -euo pipefail

mkdir -p slurm_logs

DATA_ROOT="${DATA_ROOT:-${SLURM_SUBMIT_DIR}/datasets/Field II/field_ii_data_multiangle}"
TARGET_ROOT="${TARGET_ROOT:-${SLURM_SUBMIT_DIR}/results_schiffner_targets_fieldii}"
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/results_v2/b1_v5_pulse}"
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
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.3}"
GAIN_REG_WEIGHT="${GAIN_REG_WEIGHT:-0.0}"
INIT_GAIN="${INIT_GAIN:-1.0}"
SEED="${SEED:-42}"
MAX_PHANTOMS_TRAIN="${MAX_PHANTOMS_TRAIN:-}"
MAX_PHANTOMS_VAL="${MAX_PHANTOMS_VAL:-}"
SMOKE_STEPS="${SMOKE_STEPS:-}"
RESUME="${RESUME:-0}"
NO_AUG="${NO_AUG:-0}"
FREEZE_GAIN="${FREEZE_GAIN:-0}"
PULSE_APPLY_PROB="${PULSE_APPLY_PROB:-0.7}"
PULSE_BW_MIN_HZ="${PULSE_BW_MIN_HZ:-1.0e6}"
PULSE_BW_MAX_HZ="${PULSE_BW_MAX_HZ:-2.5e6}"
PULSE_FC_SHIFT_HZ="${PULSE_FC_SHIFT_HZ:-300e3}"

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
echo "  anchor_w        : $ANCHOR_WEIGHT  gain_reg_w : $GAIN_REG_WEIGHT"
echo "  residual init   : gain=$INIT_GAIN  freeze_gain=$FREEZE_GAIN"
echo "  pulse aug       : prob=$PULSE_APPLY_PROB  bw=[${PULSE_BW_MIN_HZ},${PULSE_BW_MAX_HZ}]  fc_shift=±${PULSE_FC_SHIFT_HZ}"
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

python -u -m unet_ag.train.b1_v5_pulse_aug \
    --data_root         "$DATA_ROOT" \
    --target_root       "$TARGET_ROOT" \
    --output_dir        "$OUTPUT_DIR" \
    --phantom_type      "$PHANTOM_TYPE" \
    --epochs            "$EPOCHS" --lr "$LR" --weight_decay "$WD" \
    --nz "$NZ" --nx "$NX" \
    --base_ch           "$BASE_CH" --embed_ch "$EMBED_CH" \
    --mse_weight        "$MSE_WEIGHT" \
    --l1_weight         "$L1_WEIGHT" \
    --ssim_weight       "$SSIM_WEIGHT" \
    --anchor_weight     "$ANCHOR_WEIGHT" \
    --gain_reg_weight   "$GAIN_REG_WEIGHT" \
    --init_gain         "$INIT_GAIN" \
    --pulse_apply_prob  "$PULSE_APPLY_PROB" \
    --pulse_bw_min_hz   "$PULSE_BW_MIN_HZ" \
    --pulse_bw_max_hz   "$PULSE_BW_MAX_HZ" \
    --pulse_fc_shift_hz "$PULSE_FC_SHIFT_HZ" \
    --seed              "$SEED" \
    --device cuda \
    "${EXTRA_ARGS[@]}"

echo "=== Done — $(date) ==="
