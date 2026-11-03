#!/bin/bash
#SBATCH --job-name=v2_eval_b1v4
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/v2_eval_b1v4_%j.out
#SBATCH --error=slurm_logs/v2_eval_b1v4_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# Field II eval of the B1-v4 best.pt. Same model as B1-v3 (B1V3Residual)
# trained with the L1 anchor → reuses ``--model_type b1_v4``, which delegates
# to the b1_v3 loader in ``evaluate_fieldii.py``.

set -euo pipefail

mkdir -p slurm_logs

DATA_ROOT="${DATA_ROOT:-${SLURM_SUBMIT_DIR}/datasets/Field II/field_ii_data_multiangle}"
TARGET_ROOT="${TARGET_ROOT:-${SLURM_SUBMIT_DIR}/results_schiffner_targets_fieldii}"
CKPT="${CKPT:-${SLURM_SUBMIT_DIR}/results_v2/b1_v4/checkpoints/best.pt}"
OUT="${OUT:-${SLURM_SUBMIT_DIR}/results_v2/b1_v4/eval_fieldii.csv}"
PHANTOM_TYPE="${PHANTOM_TYPE:-both}"
NZ="${NZ:-612}"
NX="${NX:-388}"
MAX_PHANTOMS="${MAX_PHANTOMS:-}"
BASE_CH="${BASE_CH:-32}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) — $(date) ==="
echo "  ckpt          : $CKPT"
echo "  data_root     : $DATA_ROOT"
echo "  target_root   : $TARGET_ROOT"
echo "  out           : $OUT"
echo "  phantom_type  : $PHANTOM_TYPE"
echo "  grid          : ${NZ}x${NX}"
echo "  base_ch       : $BASE_CH"
[ -n "$MAX_PHANTOMS" ] && echo "  max_phantoms  : $MAX_PHANTOMS"
echo "============================================"

EXTRA_ARGS=()
[ -n "$MAX_PHANTOMS" ] && EXTRA_ARGS+=(--max_phantoms "$MAX_PHANTOMS")

python -u -m unet_ag.eval.evaluate_fieldii \
    --ckpt          "$CKPT" \
    --data_root     "$DATA_ROOT" \
    --target_root   "$TARGET_ROOT" \
    --out           "$OUT" \
    --model_type    b1_v4 \
    --base_ch       "$BASE_CH" \
    --phantom_type  "$PHANTOM_TYPE" \
    --nz "$NZ" --nx "$NX" \
    --device cuda \
    "${EXTRA_ARGS[@]}"

echo "=== Done — $(date) ==="
