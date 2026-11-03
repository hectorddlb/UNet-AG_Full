#!/bin/bash
#SBATCH --job-name=v2_a1
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00
#SBATCH --output=slurm_logs/v2_a1_%j.out
#SBATCH --error=slurm_logs/v2_a1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# Proposal A1 — RF encoder + ResU-Net + H_Fdf fidelity loss.
# Defaults: 60 epochs, MSE+L1 + λ·fidelity, AdamW, BatchNorm.

set -euo pipefail

mkdir -p slurm_logs

DATA_ROOT="${DATA_ROOT:-${SLURM_SUBMIT_DIR}/datasets/Field II/field_ii_data_multiangle}"
TARGET_ROOT="${TARGET_ROOT:-${SLURM_SUBMIT_DIR}/results_schiffner_targets_fieldii}"
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/results_v2/a1}"
PHANTOM_TYPE="${PHANTOM_TYPE:-points}"
EPOCHS="${EPOCHS:-60}"
LR="${LR:-1e-3}"
WD="${WD:-1e-4}"
# NOTE on grid (lower than B1 at 612×388):
# - A1 OOM'd at 612×388 because op.forward inside autograd graph allocates
#   ~22 GB at PICMUS grid (same root cause as Phase 4 Idea 1.2). 1 angle,
#   not 8, was insufficient to avoid OOM because Nf ~ 992 bins still
#   materialises huge kernel_b tensors per chunk in the autograd graph.
# - 256×128 ≈ 1/7 memory → fits comfortably (~3 GB peak fidelity forward).
# - Comparison vs B1 (612×388) at eval time via upsampling A1 output;
#   CNR/gCNR/FWHM are resolution-insensitive at these scales.
NZ="${NZ:-256}"
NX="${NX:-128}"
RF_N_SAMPLES="${RF_N_SAMPLES:-3500}"
BASE_CH="${BASE_CH:-32}"
L1_WEIGHT="${L1_WEIGHT:-0.5}"
FIDELITY_WEIGHT="${FIDELITY_WEIGHT:-5.0}"
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
echo "  grid            : ${NZ}x${NX}  rf_n_samples : $RF_N_SAMPLES"
echo "  base_ch         : $BASE_CH"
echo "  l1_weight       : $L1_WEIGHT  fidelity_weight : $FIDELITY_WEIGHT"
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

python -u -m unet_ag.train.a1_rf_hfdf_loss \
    --data_root     "$DATA_ROOT" \
    --target_root   "$TARGET_ROOT" \
    --output_dir    "$OUTPUT_DIR" \
    --phantom_type  "$PHANTOM_TYPE" \
    --epochs        "$EPOCHS" --lr "$LR" --weight_decay "$WD" \
    --nz "$NZ" --nx "$NX" \
    --rf_n_samples  "$RF_N_SAMPLES" \
    --base_ch       "$BASE_CH" \
    --l1_weight     "$L1_WEIGHT" \
    --fidelity_weight "$FIDELITY_WEIGHT" \
    --seed          "$SEED" \
    --device cuda \
    "${EXTRA_ARGS[@]}"

echo "=== Done — $(date) ==="
