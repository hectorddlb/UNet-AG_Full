#!/bin/bash
#SBATCH --job-name=v2_eval_a2
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/v2_eval_a2_%j.out
#SBATCH --error=slurm_logs/v2_eval_a2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# Field II evaluation of A2 best.pt — CNR/gCNR (cysts) + FWHM (points)
# with the fixed metric (NaN fallback, n_resolved counters).
# Grid defaults to A2's training grid 256×128; can override to 612×388 for
# direct visual comparison with B1 (model will upsample its sigmoid output).

set -euo pipefail

mkdir -p slurm_logs

DATA_ROOT="${DATA_ROOT:-${SLURM_SUBMIT_DIR}/datasets/Field II/field_ii_data_multiangle}"
TARGET_ROOT="${TARGET_ROOT:-${SLURM_SUBMIT_DIR}/results_schiffner_targets_fieldii}"
CKPT="${CKPT:-${SLURM_SUBMIT_DIR}/results_v2/a2/checkpoints/best.pt}"
OUT="${OUT:-${SLURM_SUBMIT_DIR}/results_v2/a2/eval_fieldii.csv}"
PHANTOM_TYPE="${PHANTOM_TYPE:-both}"
NZ="${NZ:-256}"
NX="${NX:-128}"
RF_N_SAMPLES="${RF_N_SAMPLES:-3500}"
MAX_PHANTOMS="${MAX_PHANTOMS:-}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) — $(date) ==="
echo "  ckpt          : $CKPT"
echo "  data_root     : $DATA_ROOT"
echo "  target_root   : $TARGET_ROOT"
echo "  out           : $OUT"
echo "  phantom_type  : $PHANTOM_TYPE"
echo "  grid          : ${NZ}x${NX}  rf_n_samples : $RF_N_SAMPLES"
[ -n "$MAX_PHANTOMS" ] && echo "  max_phantoms  : $MAX_PHANTOMS"
echo "============================================"

EXTRA_ARGS=()
[ -n "$MAX_PHANTOMS" ] && EXTRA_ARGS+=(--max_phantoms "$MAX_PHANTOMS")

python -u -m unet_ag.eval.evaluate_fieldii \
    --ckpt          "$CKPT" \
    --data_root     "$DATA_ROOT" \
    --target_root   "$TARGET_ROOT" \
    --out           "$OUT" \
    --model_type    a2 \
    --phantom_type  "$PHANTOM_TYPE" \
    --nz "$NZ" --nx "$NX" \
    --rf_n_samples  "$RF_N_SAMPLES" \
    --device cuda \
    "${EXTRA_ARGS[@]}"

echo "=== Done — $(date) ==="
