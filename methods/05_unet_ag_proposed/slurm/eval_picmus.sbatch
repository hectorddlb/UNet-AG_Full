#!/bin/bash
#SBATCH --job-name=v2_eval_picmus
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/v2_eval_picmus_%j.out
#SBATCH --error=slurm_logs/v2_eval_picmus_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# Eval B1-v4 (or fine-tuned variant) on PICMUS sim+exp × resolution+contrast.
# Reuses evaluate_picmus.py with --model_type b1_v4 (delegates to B1V3Residual).

set -euo pipefail

mkdir -p slurm_logs

CKPT="${CKPT:-${SLURM_SUBMIT_DIR}/results_v2/b1_v4_a0p3/checkpoints/best.pt}"
PICMUS_ROOT="${PICMUS_ROOT:-${SLURM_SUBMIT_DIR}/datasets/PICMUS}"
OUT="${OUT:-${SLURM_SUBMIT_DIR}/results_v2/b1_v4_a0p3/eval_picmus.csv}"
BASE_CH="${BASE_CH:-32}"
NZ="${NZ:-612}"
NX="${NX:-388}"
PREFILTER_FC_MHZ="${PREFILTER_FC_MHZ:-0.0}"
PREFILTER_BW_MHZ="${PREFILTER_BW_MHZ:-0.0}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) — $(date) ==="
echo "  ckpt        : $CKPT"
echo "  picmus_root : $PICMUS_ROOT"
echo "  out         : $OUT"
echo "  base_ch     : $BASE_CH  grid: ${NZ}x${NX}"
echo "  prefilter   : fc=${PREFILTER_FC_MHZ} MHz BW=${PREFILTER_BW_MHZ} MHz (0/0 disables)"
echo "============================================"

python -u -m unet_ag.eval.evaluate_picmus \
    --ckpt        "$CKPT" \
    --picmus_root "$PICMUS_ROOT" \
    --out         "$OUT" \
    --mode        both \
    --phantom     both \
    --model_type  b1_v4 \
    --base_ch     "$BASE_CH" \
    --nz "$NZ" --nx "$NX" \
    --prefilter_fc_mhz "$PREFILTER_FC_MHZ" \
    --prefilter_bw_mhz "$PREFILTER_BW_MHZ" \
    --device cuda

echo "=== Done — $(date) ==="
