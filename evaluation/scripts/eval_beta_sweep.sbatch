#!/bin/bash
#SBATCH --job-name=beta_canon
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/beta_canon_%j.out
#SBATCH --error=slurm_logs/beta_canon_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# Re-measures the beta sweep {0.3, 0.4, 0.5, 0.6, 0.7} with the canonical
# protocol (env_log 612x388 -> _eval_image), so that the beta curve is in the
# same units as the main table.
# The checkpoints live in the original repo results_v2/, outside this package.

set -euo pipefail
mkdir -p slurm_logs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"
source env.sh

echo "=== Job ${SLURM_JOB_ID} on $(hostname) — $(date) ==="
nvidia-smi || true
echo "============================================"

python evaluation/scripts/eval_beta_sweep_canonical.py --device cuda

echo "=== Done — $(date) ==="
