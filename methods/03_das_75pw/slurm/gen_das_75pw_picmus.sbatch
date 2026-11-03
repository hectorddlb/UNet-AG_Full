#!/bin/bash
#SBATCH --job-name=das75_pic
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/das75_pic_%j.out
#SBATCH --error=slurm_logs/das75_pic_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>

set -euo pipefail
mkdir -p slurm_logs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) - $(date) ==="
{ nvidia-smi 2>&1 || true; } > /tmp/nvsmi_${SLURM_JOB_ID}.txt
head -15 /tmp/nvsmi_${SLURM_JOB_ID}.txt
rm -f /tmp/nvsmi_${SLURM_JOB_ID}.txt
echo "============================================"

python unet_ag/scripts/gen_das_75pw_picmus.py \
    --picmus datasets/PICMUS \
    --out results_v2/das_75pw_picmus \
    --device cuda

echo "=== Done - $(date) ==="
