#!/bin/bash
#SBATCH --job-name=v2_ecubdl
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/v2_ecubdl_%j.out
#SBATCH --error=slurm_logs/v2_ecubdl_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# CUBDL OOD eval — evaluate_cubdl.py on the filtered subset (55 files).
# By default it goes through the 55 phantoms. CKPT/OUT can be set via sbatch --export.

set -euo pipefail

mkdir -p slurm_logs

CKPT="${CKPT:-${SLURM_SUBMIT_DIR}/results_v2/b1_v5_anchor05/checkpoints/best.pt}"
OUT="${OUT:-${SLURM_SUBMIT_DIR}/results_v2/b1_v5_anchor05/eval_cubdl.csv}"
BMODE_DIR="${BMODE_DIR:-}"
CUBDL_ROOT="${CUBDL_ROOT:-${SLURM_SUBMIT_DIR}/datasets/CUBDL}"
INSTITUTIONS="${INSTITUTIONS:-}"
MAX_PHANTOMS="${MAX_PHANTOMS:-}"
NZ="${NZ:-612}"
NX="${NX:-388}"
BASE_CH="${BASE_CH:-32}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) — $(date) ==="
echo "  ckpt        : $CKPT"
echo "  out         : $OUT"
echo "  cubdl_root  : $CUBDL_ROOT"
[ -n "$INSTITUTIONS" ] && echo "  institutions: $INSTITUTIONS"
[ -n "$MAX_PHANTOMS" ] && echo "  max_phantoms: $MAX_PHANTOMS"
[ -n "$BMODE_DIR"    ] && echo "  bmode_dir   : $BMODE_DIR"
{ nvidia-smi 2>&1 || true; } > /tmp/nvsmi_${SLURM_JOB_ID:-$$}.txt
head -15 /tmp/nvsmi_${SLURM_JOB_ID:-$$}.txt
rm -f /tmp/nvsmi_${SLURM_JOB_ID:-$$}.txt
echo "============================================"

EXTRA_ARGS=()
[ -n "$MAX_PHANTOMS" ] && EXTRA_ARGS+=(--max_phantoms "$MAX_PHANTOMS")
[ -n "$BMODE_DIR"    ] && EXTRA_ARGS+=(--save_bmode_dir "$BMODE_DIR")
if [ -n "$INSTITUTIONS" ]; then
    EXTRA_ARGS+=(--institutions)
    for inst in $INSTITUTIONS; do
        EXTRA_ARGS+=("$inst")
    done
fi

python -u -m unet_ag.eval.evaluate_cubdl \
    --ckpt        "$CKPT" \
    --cubdl_root  "$CUBDL_ROOT" \
    --out         "$OUT" \
    --nz          "$NZ" \
    --nx          "$NX" \
    --base_ch     "$BASE_CH" \
    --device      cuda \
    "${EXTRA_ARGS[@]}"

echo "=== Done — $(date) ==="
