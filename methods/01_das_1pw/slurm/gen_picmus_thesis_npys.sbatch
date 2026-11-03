#!/bin/bash
#SBATCH --job-name=pic_thesis
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/pic_thesis_%j.out
#SBATCH --error=slurm_logs/pic_thesis_%j.err
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

python unet_ag/scripts/gen_picmus_npys_for_thesis.py \
    --picmus datasets/PICMUS \
    --out_das results_v2/das_1pw_picmus \
    --out_b1 results_v2/b1_v5_anchor05_inference \
    --ckpt results_v2/b1_v5_anchor05/checkpoints/best.pt \
    --device cuda

echo "=== Done - $(date) ==="
