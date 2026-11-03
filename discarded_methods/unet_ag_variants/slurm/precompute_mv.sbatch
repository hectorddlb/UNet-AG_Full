#!/bin/bash
#SBATCH --job-name=v2_mv
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/v2_mv_%A_%a.out
#SBATCH --error=slurm_logs/v2_mv_%A_%a.err
#SBATCH --array=0-39%4
#
# Precompute MV 1 PW baselines. Per-pixel matrix inversion dominates.
# Grid reduced to 256×128 + subarray_len=32 → ~8× cheaper than the original
# 384×256 + L=64. Used only as one of three supervisor targets in B2; eval
# benchmark MV computed at full PICMUS res in evaluate_*.py on demand.
# 15 phantoms/task × ~20 min/phantom = ~5h/task, fits in 6h.
# 40 tasks × 1 concurrent (QOS) ≈ 8 days wallclock — still long but at least
# it can run in background; alt: skip MV for now and use only Schiffner+DAS75.

set -euo pipefail

mkdir -p slurm_logs

DATA_ROOT="${DATA_ROOT:-${SLURM_SUBMIT_DIR}/datasets/Field II/field_ii_data_multiangle}"
OUT_ROOT="${OUT_ROOT:-${SLURM_SUBMIT_DIR}/results_v2/mv}"
PHANTOM_TYPE="${PHANTOM_TYPE:-both}"
NZ="${NZ:-256}"
NX="${NX:-128}"
SUBARRAY_LEN="${SUBARRAY_LEN:-32}"
DIAG_LOAD="${DIAG_LOAD:-1e-3}"
F_NUMBER="${F_NUMBER:-1.5}"
ARRAY_TOTAL="${SLURM_ARRAY_TASK_COUNT:-40}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"

echo "=== Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}/${ARRAY_TOTAL} on $(hostname) — $(date) ==="

python -m unet_ag.scripts.precompute_mv \
    --data_root "$DATA_ROOT" \
    --out_root  "$OUT_ROOT" \
    --phantom_type "$PHANTOM_TYPE" \
    --nz "$NZ" --nx "$NX" \
    --subarray_len "$SUBARRAY_LEN" \
    --diag_load "$DIAG_LOAD" \
    --f_number "$F_NUMBER" \
    --array_idx "$SLURM_ARRAY_TASK_ID" \
    --array_total "$ARRAY_TOTAL"

echo "=== Done — $(date) ==="
