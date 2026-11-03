#!/bin/bash
#SBATCH --job-name=v2_prefilter
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/v2_prefilter_%j.out
#SBATCH --error=slurm_logs/v2_prefilter_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# (B) — OOD penalty diagnosis: 5×5 grid search over (fc, BW) of a Gaussian
# bandpass applied to the PICMUS RF before the adjoint in B1-v4(0.3).
#
# Empirical measurement (2026-05-29) of the Field II ↔ PICMUS spectral gap:
#   Field II training:        fc_peak=4.97 MHz   BW(-6dB)=1.82 MHz
#   PICMUS exp resolution:    fc_peak=5.04 MHz   BW(-6dB)=1.04 MHz
#
# If some (fc, BW) ≠ identity gives val_total < identity on the exp phantoms
# with a margin ≥3% → spectral gap confirmed → defensible as an ablation and
# as motivation for (A) pulse-shape augmentation during full training (B1-v5).
# If it converges to the same val_total over the whole grid → it needs an
# intervention during full training. (A) remains the mandatory route.
#
# Estimated cost: 104 evals (4 phantoms × 26 pares (fc,BW,id)) ≈ 10-15 min.

set -euo pipefail

mkdir -p slurm_logs

CKPT="${CKPT:-${SLURM_SUBMIT_DIR}/results_v2/b1_v4_a0p3/checkpoints/best.pt}"
PICMUS_ROOT="${PICMUS_ROOT:-${SLURM_SUBMIT_DIR}/datasets/PICMUS}"
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/results_v2/b1_v4_a0p3_prefilter_grid}"
NZ="${NZ:-612}"
NX="${NX:-388}"
BASE_CH="${BASE_CH:-32}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) — $(date) ==="
echo "  ckpt        : $CKPT"
echo "  picmus_root : $PICMUS_ROOT"
echo "  output_dir  : $OUTPUT_DIR"
echo "  grid        : ${NZ}x${NX}"
{ nvidia-smi 2>&1 || true; } > /tmp/nvsmi_${SLURM_JOB_ID:-$$}.txt
head -15 /tmp/nvsmi_${SLURM_JOB_ID:-$$}.txt
rm -f /tmp/nvsmi_${SLURM_JOB_ID:-$$}.txt
echo "============================================"

python -u -m unet_ag.scripts.diagnose_prefilter_picmus \
    --ckpt        "$CKPT" \
    --picmus_root "$PICMUS_ROOT" \
    --output_dir  "$OUTPUT_DIR" \
    --nz          "$NZ" \
    --nx          "$NX" \
    --base_ch     "$BASE_CH" \
    --device      cuda

echo "=== Done — $(date) ==="
