#!/bin/bash
#SBATCH --job-name=final_results
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/final_results_%j.out
#SBATCH --error=slurm_logs/final_results_%j.err
#
# Full chain of the thesis evaluation protocol.
# Launch from the repository root:
#   THESIS_DATASETS=/path/datasets DRUS_ARTIFACTS=/path/artifacts \
#       sbatch evaluation/scripts/run_final_results.sbatch
# Without the DRUS artifacts, step 3 is skipped and the evaluation uses the
# evaluation/npys/drus/ files that already exist.

set -euo pipefail
mkdir -p slurm_logs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"
source env.sh

echo "=== Job ${SLURM_JOB_ID} on $(hostname) — $(date) ==="
nvidia-smi | head -12 || true
echo "============================================"

echo "--- [1/4] Classical (DAS and Fd-F, 1 and 75 PW), corrected transmit convention"
python evaluation/scripts/gen_classical_picmus_fixed.py --device cuda

echo "--- [2/4] UNet-AG (B1-v5_anchor05), 3 seeds"
python evaluation/scripts/gen_b1v5_picmus_seeds.py --device cuda

echo "--- [3/4] DRUS (prior without fine-tuning, thesis γ, seed 0)"
if python -m drus.cli info >/dev/null 2>&1; then
    python evaluation/scripts/gen_drus_picmus.py
else
    echo "    DRUS artifacts not available: skipped (see methods/04_drus/README.md)"
fi

echo "--- [4/4] Unified evaluation + tables"
python evaluation/scripts/eval_and_figs_final.py

echo "=== Done — $(date) ==="
