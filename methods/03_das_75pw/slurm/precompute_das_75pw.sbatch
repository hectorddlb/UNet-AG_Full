#!/bin/bash
#SBATCH --job-name=v2_das75
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/v2_das75_%A_%a.out
#SBATCH --error=slurm_logs/v2_das75_%A_%a.err
#SBATCH --array=0-59%4
#
# Precompute DAS 75 PW compounded targets for the Field II multi-angle dataset.
# CPU-only — runs on a GPU node but doesn't touch the GPU.
# Grid reduced to 300×200 (vs PICMUS 612×388) — DAS75 is a supervisor target
# for B2, not the eval baseline. Upsampled bilinearly at training time. ~2 min
# per phantom × 10 phantoms/task = ~20 min/task, fits comfortably in 6h.
# 60 tasks × 1 concurrent (QOS) ≈ 20 h wallclock total.

set -euo pipefail

mkdir -p slurm_logs

DATA_ROOT="${DATA_ROOT:-${SLURM_SUBMIT_DIR}/datasets/Field II/field_ii_data_multiangle}"
OUT_ROOT="${OUT_ROOT:-${SLURM_SUBMIT_DIR}/results_v2/das75}"
PHANTOM_TYPE="${PHANTOM_TYPE:-both}"
NZ="${NZ:-300}"
NX="${NX:-200}"
F_NUMBER="${F_NUMBER:-1.5}"
ARRAY_TOTAL="${SLURM_ARRAY_TASK_COUNT:-60}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"

echo "=== Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}/${ARRAY_TOTAL} on $(hostname) — $(date) ==="
echo "  data_root     : $DATA_ROOT"
echo "  out_root      : $OUT_ROOT"
echo "  phantom_type  : $PHANTOM_TYPE"
echo "  grid          : ${NZ}x${NX}  f_number=${F_NUMBER}"

python -m unet_ag.scripts.precompute_das_75pw \
    --data_root "$DATA_ROOT" \
    --out_root  "$OUT_ROOT" \
    --phantom_type "$PHANTOM_TYPE" \
    --nz "$NZ" --nx "$NX" \
    --f_number "$F_NUMBER" \
    --array_idx "$SLURM_ARRAY_TASK_ID" \
    --array_total "$ARRAY_TOTAL"

echo "=== Done — $(date) ==="
