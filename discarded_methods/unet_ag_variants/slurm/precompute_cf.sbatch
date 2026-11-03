#!/bin/bash
#SBATCH --job-name=v2_cf
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/v2_cf_%A_%a.out
#SBATCH --error=slurm_logs/v2_cf_%A_%a.err
#SBATCH --array=0-9%4
#
# Precompute Coherence Factor 1 PW baselines. Cheap.

set -euo pipefail

mkdir -p slurm_logs

DATA_ROOT="${DATA_ROOT:-${SLURM_SUBMIT_DIR}/datasets/Field II/field_ii_data_multiangle}"
OUT_ROOT="${OUT_ROOT:-${SLURM_SUBMIT_DIR}/results_v2/cf}"
PHANTOM_TYPE="${PHANTOM_TYPE:-both}"
NZ="${NZ:-612}"
NX="${NX:-388}"
F_NUMBER="${F_NUMBER:-1.5}"
ARRAY_TOTAL="${SLURM_ARRAY_TASK_COUNT:-10}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"

echo "=== Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}/${ARRAY_TOTAL} on $(hostname) — $(date) ==="

python -m unet_ag.scripts.precompute_cf \
    --data_root "$DATA_ROOT" \
    --out_root  "$OUT_ROOT" \
    --phantom_type "$PHANTOM_TYPE" \
    --nz "$NZ" --nx "$NX" \
    --f_number "$F_NUMBER" \
    --array_idx "$SLURM_ARRAY_TASK_ID" \
    --array_total "$ARRAY_TOTAL"

echo "=== Done — $(date) ==="
